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

"""Numerical coverage of independently selected narrowing-prefill stages.

A tiny tensor model supplies weights instead of loading a checkpoint. The
runner, input buffers and CUDA captures are real. Its eager attention breaks
write a cache on every source row and consume live request-window metadata;
the comparison checks outputs, taps and cache writes, not mock call order.
DP metadata is replicated locally here; distributed collective completion is
covered by the engine tests rather than a simulated process group.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=15, suite="runtime-1gpu")

from tokenspeed.runtime.execution.breakable_cuda_graph import (  # noqa: E402
    break_point,
    current_forward_ctx,
)
from tokenspeed.runtime.execution.context import ForwardContext  # noqa: E402
from tokenspeed.runtime.execution.forward_batch_info import (  # noqa: E402
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.execution.input_buffer import InputBuffers  # noqa: E402
from tokenspeed.runtime.execution.memory_delta import (  # noqa: E402
    NULL_MEMORY_DELTA_OBSERVER,
)
from tokenspeed.runtime.execution.model_executor import (  # noqa: E402
    ModelExecutorConfig,
)
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout  # noqa: E402
from tokenspeed.runtime.execution.prefill_graph import PrefillGraph  # noqa: E402
from tokenspeed.runtime.multimodal.inputs import MultimodalForwardContext  # noqa: E402
from tokenspeed.runtime.utils.torch_memory_saver_adapter import (  # noqa: E402
    TorchMemorySaverAdapter,
)


@dataclass(frozen=True)
class WindowRowState:
    hidden: torch.Tensor
    positions: torch.Tensor
    tap: torch.Tensor

    @property
    def rows(self) -> int:
        return self.hidden.shape[0]

    def land_into(self, target: WindowRowState) -> None:
        if self.rows > target.rows:
            raise ValueError("narrowed state exceeds the decoder capacity")
        for source, destination in (
            (self.hidden, target.hidden),
            (self.positions, target.positions),
            (self.tap, target.tap),
        ):
            destination[: self.rows].copy_(source)
            destination[self.rows :].zero_()


class WindowBackend:
    """Request-window metadata used by the tensor model's eager operators."""

    def __init__(self, window: int) -> None:
        self.window = window
        self.step_counter = None
        self.source_rows = 0
        self.kept_rows = torch.empty(0, dtype=torch.int64, device="cuda")
        self.logits_rows = torch.empty_like(self.kept_rows)

    def init_prefill_graph_state(self, *, max_num_tokens: int, max_bs: int) -> None:
        pass

    def prepare_prefill_metadata(self, *args, capture: bool) -> bool:
        return False

    def init_forward_metadata(
        self,
        *,
        extend_seq_lens_cpu: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_prompt_lens_cpu: torch.Tensor,
        num_extends: int,
        **kwargs,
    ) -> None:
        counts = extend_seq_lens_cpu[:num_extends].tolist()
        prefixes = extend_prefix_lens_cpu[:num_extends].tolist()
        prompts = extend_prompt_lens_cpu[:num_extends].tolist()
        completes = [
            prefix + count == prompt
            for count, prefix, prompt in zip(counts, prefixes, prompts, strict=True)
        ]
        any_outputs = any(completes)
        kept, output_rows = [], []
        start = 0
        for count, complete in zip(counts, completes, strict=True):
            width = min(count, self.window if complete else int(any_outputs))
            kept.extend(range(start + count - width, start + count))
            if complete:
                output_rows.append(len(kept) - 1)
            start += count
        self.source_rows = start
        self.kept_rows = torch.tensor(kept, dtype=torch.int64, device="cuda")
        self.logits_rows = torch.tensor(output_rows, dtype=torch.int64, device="cuda")


class WindowInputs(InputBuffers):
    """Real input buffers with an additional per-row model input."""

    def __init__(self) -> None:
        super().__init__(
            max_bs=8,
            max_num_tokens=512,
            state_write_padding_pool_index=9,
            device="cuda",
            reads_state_checkpoints=False,
        )
        self.row_features = torch.zeros(512, 8, device="cuda")

    def ngram_model_kwargs(self, num_tokens: int) -> dict[str, torch.Tensor]:
        return {"row_features": self.row_features[:num_tokens]}


class WindowTransformer(torch.nn.Module):
    """One token-local stage, a window narrowing, and a decoder stage."""

    def __init__(self) -> None:
        super().__init__()
        self.max_decoder_rows_per_request = 8
        self.embed_tokens = torch.nn.Embedding(128, 8, device="cuda")
        with torch.no_grad():
            self.embed_tokens.weight.copy_(
                torch.arange(1024, device="cuda").reshape(128, 8) / 1024
            )
        self.register_buffer("encoder_cache", torch.zeros(512, 8, device="cuda"))

    def prepare_model_kwargs(self, ctx, input_ids, kwargs):
        return dict(kwargs)

    def encoder_forward(
        self, input_ids, positions, ctx, *, input_embeds, row_features
    ) -> WindowRowState:
        projected = input_embeds * 2 + positions[:, None] / 16 + row_features
        hidden = self.encoder_attention(projected, ctx)
        return WindowRowState(hidden, positions, hidden * 5)

    @break_point
    def encoder_attention(self, projected, ctx):
        rows = ctx.attn_backend.source_rows
        self.encoder_cache[:rows].copy_(projected[:rows])
        return projected[:rows] + 1

    def narrowing_forward(self, state, ctx) -> WindowRowState:
        keep = ctx.attn_backend.kept_rows
        return WindowRowState(
            state.hidden.index_select(0, keep) + 2,
            state.positions.index_select(0, keep),
            state.tap.index_select(0, keep),
        )

    @break_point
    def decoder_attention(self, projected, positions, ctx):
        rows = ctx.attn_backend.kept_rows.numel()
        return projected[:rows].relu() + positions[:rows, None] / 8

    def decoder_forward(self, state, ctx):
        hidden = self.decoder_attention(state.hidden * 3, state.positions, ctx) * 4
        return hidden, [state.tap * 7]

    def finish_forward(self, hidden, captured, ctx):
        rows = ctx.attn_backend.logits_rows
        ctx.logits_rows_selected = True
        return hidden.index_select(0, rows), [
            tap.index_select(0, rows) for tap in captured
        ]

    def decoder_rows(self, ctx) -> int:
        return ctx.attn_backend.kept_rows.numel()

    def allocate_decoder_state(self, rows: int) -> WindowRowState:
        return WindowRowState(
            torch.zeros(rows, 8, device="cuda"),
            torch.zeros(rows, dtype=torch.int64, device="cuda"),
            torch.zeros(rows, 8, device="cuda"),
        )

    def forward(self, input_ids, positions, ctx, **kwargs):
        encoded = self.encoder_forward(input_ids, positions, ctx, **kwargs)
        narrowed = self.narrowing_forward(encoded, ctx)
        hidden, captured = self.decoder_forward(narrowed, ctx)
        return self.finish_forward(hidden, captured, ctx)


class WindowReadout(torch.nn.Module):
    def forward(self, input_ids, hidden, head, metadata, auxiliary):
        return head(hidden), auxiliary


class WindowCausalLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = WindowTransformer()
        self.lm_head = torch.nn.Linear(8, 4, bias=False, device="cuda")
        with torch.no_grad():
            self.lm_head.weight.copy_(
                torch.arange(32, device="cuda").reshape(4, 8) / 32
            )
        self.logits_processor = WindowReadout()


@dataclass(frozen=True)
class MaskedRowState(WindowRowState):
    image_mask: torch.Tensor

    def land_into(self, target: MaskedRowState) -> None:
        super().land_into(target)
        target.image_mask[: self.rows].copy_(self.image_mask)
        target.image_mask[self.rows :].zero_()


class MaskedTransformer(WindowTransformer):
    """A modality-dependent token path on both sides of the narrowing."""

    def __init__(self) -> None:
        super().__init__()
        self.max_decoder_rows_per_request = 128

    def encoder_forward(
        self, input_ids, positions, ctx, *, input_embeds, row_features, image_mask
    ) -> MaskedRowState:
        encoded = super().encoder_forward(
            input_ids,
            positions,
            ctx,
            input_embeds=input_embeds + image_mask[:, None] * 11,
            row_features=row_features,
        )
        return MaskedRowState(
            encoded.hidden, encoded.positions, encoded.tap, image_mask
        )

    def narrowing_forward(self, state, ctx) -> MaskedRowState:
        narrowed = super().narrowing_forward(state, ctx)
        return MaskedRowState(
            narrowed.hidden,
            narrowed.positions,
            narrowed.tap,
            state.image_mask.index_select(0, ctx.attn_backend.kept_rows),
        )

    def decoder_forward(self, state, ctx):
        hidden, captured = super().decoder_forward(state, ctx)
        return hidden + state.image_mask[:, None] * 17, captured

    def allocate_decoder_state(self, rows: int) -> MaskedRowState:
        state = super().allocate_decoder_state(rows)
        return MaskedRowState(
            state.hidden,
            state.positions,
            state.tap,
            torch.zeros(rows, dtype=torch.bool, device="cuda"),
        )


class MaskedCausalLM(WindowCausalLM):
    def __init__(self) -> None:
        super().__init__()
        self.model = MaskedTransformer()

    def allocate_prefill_graph_inputs(self, num_tokens):
        return {"image_mask": torch.zeros(num_tokens, dtype=torch.bool, device="cuda")}

    def prepare_prefill_graph_inputs(
        self, input_ids, ctx, *, model_kwargs, multimodal_context
    ):
        result = dict(model_kwargs)
        if "image_mask" not in result:
            result["image_mask"] = torch.zeros_like(input_ids, dtype=torch.bool)
        return result


@dataclass(frozen=True)
class ModelCapabilities:
    requires_request_token_history: bool
    is_multimodal_active: bool


@dataclass(frozen=True)
class ModelRunner:
    model: WindowCausalLM
    model_config: ModelCapabilities
    is_generation: bool


@dataclass(frozen=True)
class CacheArena:
    cache_group_specs: tuple


@dataclass(frozen=True)
class CachePool:
    arena: CacheArena
    layerwise_load_tracker: None


class StageHarness:
    def __init__(
        self,
        dp_size: int,
        *,
        encoder_sizes: list[int],
        decoder_sizes: list[int] | None,
        model: WindowCausalLM | None = None,
    ) -> None:
        self.dp_size = dp_size
        self.inputs = WindowInputs()
        self.model = (WindowCausalLM() if model is None else model).eval()
        self.backend = WindowBackend(self.model.model.max_decoder_rows_per_request)
        self.pool = CachePool(CacheArena(()), None)
        self.config = ModelExecutorConfig(
            enable_dp_local_cuda_graph=False,
            max_req_pool_size=9,
            output_length=16,
            enforce_eager=False,
            prefix_granularity=1,
            max_num_seqs=8 * dp_size,
            chunked_prefill_size=512,
            vocab_size=128,
            input_vocab_size=128,
            context_len=512,
            physical_context_len=512,
            device="cuda",
            gpu_id=0,
            global_rank=1,
            cudagraph_capture_sizes=None,
            disable_cuda_graph_padding=False,
            spec_topk=1,
            max_cudagraph_capture_size=4,
            model_is_mrope=False,
            autotune_cache_key=None,
            prefill_only=False,
            decode_only_attention=False,
            prefill_graph_capture_batch_sizes=None,
            prefill_graph_decoder_capture_sizes=decoder_sizes,
            input_logprob_chunk_tokens=128,
            enable_speculative_sampling=False,
            query_shard_size=1,
            query_shard_rank=0,
            enable_memory_saver=False,
            data_parallel_size=dp_size,
            world_size=dp_size,
            prefill_graph_max_tokens=max(encoder_sizes),
            prefill_graph_capture_sizes=encoder_sizes,
        )
        self.graph = PrefillGraph(
            model_runner=ModelRunner(
                self.model,
                ModelCapabilities(False, isinstance(self.model, MaskedCausalLM)),
                True,
            ),
            attn_backend=self.backend,
            token_to_kv_pool=self.pool,
            input_buffers=self.inputs,
            config=self.config,
            memory_saver_adapter=TorchMemorySaverAdapter.create(enable=False),
            num_warmup=2,
        )
        self.graph.capture(entries=None, observer=NULL_MEMORY_DELTA_OBSERVER)

    def prepare(self, counts, *, completes, prefix: int, trial: int):
        count = sum(counts)
        bs = len(counts)
        self.inputs.input_ids_buf[:count].copy_(
            (torch.arange(count, device="cuda") + trial * 13) % 127
        )
        positions = torch.cat(
            [torch.arange(prefix, prefix + width, device="cuda") for width in counts]
        )
        self.inputs.positions_buf[:count].copy_(positions)
        self.inputs.row_features[:count].copy_(
            torch.arange(count * 8, device="cuda").reshape(count, 8) / 4096 + trial
        )
        self.inputs.extend_seq_lens_cpu[:bs].copy_(torch.tensor(counts))
        self.inputs.extend_prefix_lens_cpu[:bs].fill_(prefix)
        self.inputs.extend_prompt_lens_cpu[:bs].copy_(
            torch.tensor(
                [
                    prefix + width + (0 if done else 16)
                    for width, done in zip(counts, completes)
                ]
            )
        )
        self.backend.init_forward_metadata(
            extend_seq_lens_cpu=self.inputs.extend_seq_lens_cpu[:bs],
            extend_prefix_lens_cpu=self.inputs.extend_prefix_lens_cpu[:bs],
            extend_prompt_lens_cpu=self.inputs.extend_prompt_lens_cpu[:bs],
            num_extends=bs,
        )
        ctx = ForwardContext(
            attn_backend=self.backend,
            token_to_kv_pool=self.pool,
            bs=bs,
            num_extends=bs,
            input_num_tokens=count,
            forward_mode=ForwardMode.EXTEND,
            output_layout=ForwardOutputLayout(bs, sum(completes), 0, 1),
            global_num_tokens=[count] * self.dp_size if self.dp_size > 1 else None,
            global_bs=[bs] * self.dp_size if self.dp_size > 1 else None,
            all_extend=True,
        )
        return (
            ctx,
            self.inputs.input_ids_buf[:count],
            positions,
            self.inputs.ngram_model_kwargs(count),
        )

    def eager(self, ctx, ids, positions, kwargs):
        kwargs = dict(kwargs)
        if "input_embeds" not in kwargs:
            kwargs["input_embeds"] = self.model.model.embed_tokens(ids)
        hidden, auxiliary = self.model.model(
            ids,
            positions,
            ctx,
            **kwargs,
        )
        return self.model.lm_head(hidden), auxiliary


@pytest.fixture(params=[1, 4], ids=["dp1", "dp4-metadata"])
def stages(request):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    with torch.inference_mode():
        yield StageHarness(
            request.param,
            encoder_sizes=[8, 16, 32],
            decoder_sizes=[8, 16, 32],
        )


def test_explicit_decoder_capture_ladder_is_independent_of_encoder_rungs():
    from tokenspeed.runtime.execution.prefill_graph import get_decoder_row_buckets

    requested = [8, 16, 24, 48, 128]
    assert get_decoder_row_buckets([8, 64], 8, 8, capture_sizes=requested) == [
        8,
        16,
        24,
        48,
        64,
    ]
    assert get_decoder_row_buckets([16, 32, 64], 8, 8, capture_sizes=requested) == [
        8,
        16,
        24,
        48,
        64,
    ]
    assert get_decoder_row_buckets([8, 32], 8, 3, capture_sizes=requested) == [
        8,
        16,
        24,
    ]
    assert get_decoder_row_buckets([], 8, 8, capture_sizes=requested) == []


def test_omitted_decoder_capture_ladder_preserves_existing_buckets():
    from tokenspeed.runtime.execution.prefill_graph import get_decoder_row_buckets

    assert get_decoder_row_buckets([8, 16, 64], 8, 8) == [8, 16, 64]
    assert get_decoder_row_buckets([8, 16, 64], 8, 3) == [8, 16, 24]


def test_cli_carries_two_capture_lists_under_one_token_cap():
    from tokenspeed.cli._argsplit import split_argv
    from tokenspeed.runtime.execution.prefill_graph import (
        get_decoder_row_buckets,
        get_prefill_token_buckets,
    )
    from tokenspeed.runtime.utils.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    argv = [
        "--model",
        "test",
        "--prefill-graph-max-tokens",
        "32",
        "--prefill-graph-capture-sizes",
        "8",
        "32",
        "--prefill-graph-decoder-capture-sizes",
        "8",
        "16",
        "24",
        "64",
    ]
    for args in (argv, split_argv(argv).engine):
        config = parser.parse_args(args)
        encoder = get_prefill_token_buckets(config)
        assert encoder == [8, 32]
        assert get_decoder_row_buckets(
            encoder, 8, 8, capture_sizes=config.prefill_graph_decoder_capture_sizes
        ) == [8, 16, 24, 32]


def test_distinct_encoder_and_decoder_capture_ladders_replay_numerically():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    with torch.inference_mode():
        owner = StageHarness(4, encoder_sizes=[8, 32], decoder_sizes=[8, 16, 24, 32])
        for trial, count in enumerate((5, 6, 4)):
            ctx, ids, positions, kwargs = owner.prepare(
                [count] * 3, completes=[True] * 3, prefix=0, trial=trial
            )
            expected, expected_aux = owner.eager(ctx, ids, positions, kwargs)
            expected = expected.clone()
            expected_aux = [value.clone() for value in expected_aux]
            actual, auxiliary = owner.graph.replay(
                ctx, ids, positions=positions, model_kwargs=kwargs
            )
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            for actual_tap, expected_tap in zip(auxiliary, expected_aux, strict=True):
                torch.testing.assert_close(
                    actual_tap, expected_tap, atol=1e-5, rtol=1e-5
                )
            # This output belongs to decoder[24], a capacity the encoder never
            # captured. Compare its emitting rows to this forward's eager taps.
            decoder = owner.graph._decoders[24]
            emitting = owner.backend.logits_rows
            hidden = decoder.output.hidden_states.index_select(0, emitting)
            torch.testing.assert_close(
                owner.model.lm_head(hidden), expected, atol=1e-5, rtol=1e-5
            )


@pytest.mark.parametrize("count", [15, 100], ids=["encoder-graph", "encoder-eager"])
def test_explicit_input_embeddings_survive_either_encoder_route(stages, count):
    with torch.inference_mode():
        ctx, ids, positions, kwargs = stages.prepare(
            [count], completes=[True], prefix=0, trial=1
        )
        kwargs["input_embeds"] = (
            torch.arange(count * 8, device="cuda").reshape(count, 8) / 32 + 11
        )
        expected, expected_aux = stages.eager(ctx, ids, positions, kwargs)
        expected = expected.clone()
        expected_aux = [value.clone() for value in expected_aux]
        actual, auxiliary = stages.graph.replay(
            ctx, ids, positions=positions, model_kwargs=kwargs
        )
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        for actual_tap, expected_tap in zip(auxiliary, expected_aux, strict=True):
            torch.testing.assert_close(actual_tap, expected_tap, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("count", [15, 100], ids=["encoder-graph", "encoder-eager"])
@pytest.mark.parametrize("overlap", [False, True], ids=["new-storage", "overlap"])
def test_explicit_row_inputs_reach_encoder_and_cache(stages, count, overlap):
    with torch.inference_mode():
        ctx, ids, positions, kwargs = stages.prepare(
            [count], completes=[True], prefix=16, trial=1
        )
        kwargs["row_features"] = (
            stages.inputs.row_features[1 : count + 1]
            if overlap
            else kwargs["row_features"] + 1
        )
        stages.model.model.encoder_cache.fill_(-991)
        expected, expected_aux = stages.eager(ctx, ids, positions, kwargs)
        expected = expected.clone()
        expected_aux = [value.clone() for value in expected_aux]
        expected_cache = stages.model.model.encoder_cache.clone()
        stages.model.model.encoder_cache.fill_(-991)
        actual, auxiliary = stages.graph.replay(
            ctx, ids, positions=positions, model_kwargs=kwargs
        )
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        for value, reference in zip(auxiliary, expected_aux, strict=True):
            torch.testing.assert_close(value, reference, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(stages.model.model.encoder_cache, expected_cache)


@pytest.mark.parametrize("count", [15, 16], ids=["padded", "exact"])
@pytest.mark.parametrize("start", [0, 1], ids=["same-address", "shifted-address"])
def test_strided_position_override_can_overlap_static_inputs(stages, count, start):
    with torch.inference_mode():
        ctx, ids, _, kwargs = stages.prepare(
            [count], completes=[True], prefix=0, trial=1
        )
        stages.inputs.positions_buf[: 2 * count + start].copy_(
            torch.arange(2 * count + start, device="cuda")
        )
        positions = stages.inputs.positions_buf[start : 2 * count + start : 2]
        stages.model.model.encoder_cache.fill_(-991)
        expected, expected_aux = stages.eager(ctx, ids, positions, kwargs)
        expected = expected.clone()
        expected_aux = [value.clone() for value in expected_aux]
        expected_cache = stages.model.model.encoder_cache.clone()
        stages.model.model.encoder_cache.fill_(-991)
        actual, auxiliary = stages.graph.replay(
            ctx, ids, positions=positions, model_kwargs=kwargs
        )
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        for value, reference in zip(auxiliary, expected_aux, strict=True):
            torch.testing.assert_close(value, reference, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(stages.model.model.encoder_cache, expected_cache)


def test_decoder_replays_after_encoder_input_exceeds_its_largest_bucket(stages):
    """The full real encoder span reaches cache and the decoder static state."""
    with torch.inference_mode():
        for trial, count in enumerate((113, 79, 200)):
            ctx, ids, positions, kwargs = stages.prepare(
                [count], completes=[True], prefix=16, trial=trial
            )
            # An explicit override is outside the encoder's graph input buffer.
            positions = positions + 7
            stages.model.model.encoder_cache.fill_(-991)
            expected, expected_aux = stages.eager(ctx, ids, positions, kwargs)
            expected = expected.clone()
            expected_aux = [value.clone() for value in expected_aux]
            expected_cache = stages.model.model.encoder_cache.clone()
            stages.model.model.encoder_cache.fill_(-991)

            assert stages.graph.can_run(
                ctx
            ), "the decoder has a fitting captured bucket"
            actual, auxiliary = stages.graph.replay(
                ctx, ids, positions=positions, model_kwargs=kwargs
            )
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            for actual_tap, expected_tap in zip(auxiliary, expected_aux, strict=True):
                torch.testing.assert_close(
                    actual_tap, expected_tap, atol=1e-5, rtol=1e-5
                )
            torch.testing.assert_close(stages.model.model.encoder_cache, expected_cache)
            assert stages.graph._input_embeds_buf.shape[0] == 32
            assert ctx.input_num_tokens == count
            assert ctx.global_num_tokens == (
                [count] * stages.dp_size if stages.dp_size > 1 else None
            )
            assert ctx.global_bs == (
                [1] * stages.dp_size if stages.dp_size > 1 else None
            )
            assert current_forward_ctx() is None


def test_encoder_and_decoder_graphs_refresh_across_large_and_small_batches(stages):
    """Switching the encoder route keeps new inputs, taps and padding fresh."""
    with torch.inference_mode():
        for trial, counts in enumerate(([90, 21], [15], [8, 6], [130])):
            ctx, ids, positions, kwargs = stages.prepare(
                counts, completes=[True] * len(counts), prefix=0, trial=trial
            )
            expected, expected_aux = stages.eager(ctx, ids, positions, kwargs)
            expected = expected.clone()
            expected_aux = [value.clone() for value in expected_aux]
            actual, auxiliary = stages.graph.replay(
                ctx, ids, positions=positions, model_kwargs=kwargs
            )
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            for actual_tap, expected_tap in zip(auxiliary, expected_aux, strict=True):
                torch.testing.assert_close(
                    actual_tap, expected_tap, atol=1e-5, rtol=1e-5
                )


def test_zero_live_decoder_rows_still_clear_dp_replay_state(stages):
    if stages.dp_size == 1:
        pytest.skip("DP1 skips a decoder with no live rows")
    with torch.inference_mode():
        for trial, complete in enumerate((True, False, True)):
            ctx, ids, positions, kwargs = stages.prepare(
                [111], completes=[complete], prefix=32, trial=trial
            )
            # Peers can still have completing requests while this rank has an
            # unfinished chunk; request counts and selected capacities match.
            ctx.global_num_tokens = [111, 80, 101, 73]
            ctx.global_bs = [1, 1, 1, 1]
            expected, _ = stages.eager(ctx, ids, positions, kwargs)
            expected = expected.clone()
            assert stages.graph.can_run(ctx)
            actual, auxiliary = stages.graph.replay(
                ctx, ids, positions=positions, model_kwargs=kwargs
            )
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            assert ctx.input_num_tokens == 111
            assert ctx.global_num_tokens == [111, 80, 101, 73]
            assert ctx.global_bs == [1, 1, 1, 1]
            if not complete:
                assert actual.shape == (0, 4)
                assert auxiliary[0].shape == (0, 8)
                torch.testing.assert_close(
                    stages.graph._decoders[8].statics.hidden,
                    torch.zeros_like(stages.graph._decoders[8].statics.hidden),
                )


def test_decoder_only_admission_preserves_mode_and_dp_guards(stages):
    with torch.inference_mode():
        ctx, _, _, _ = stages.prepare([100], completes=[True], prefix=0, trial=0)
        assert stages.graph.can_run(ctx)
        ctx.forward_mode = ForwardMode.MIXED
        assert not stages.graph.can_run(ctx)
        ctx.forward_mode = ForwardMode.EXTEND
        ctx.capture_hidden_mode = CaptureHiddenMode.FULL
        assert not stages.graph.can_run(ctx)
        ctx.capture_hidden_mode = CaptureHiddenMode.NULL
        if stages.dp_size > 1:
            ctx.all_extend = False
            assert not stages.graph.can_run(ctx)
            ctx.all_extend = True
            ctx.global_bs = None
            assert not stages.graph.can_run(ctx)
            ctx.global_bs = [1] * stages.dp_size
            ctx.global_num_tokens = None
            assert not stages.graph.can_run(ctx)


def test_no_graph_admission_when_both_stage_capacities_are_exceeded(stages):
    with torch.inference_mode():
        ctx, _, _, _ = stages.prepare([21] * 5, completes=[True] * 5, prefix=0, trial=0)
        assert not stages.graph.can_run(ctx)


def test_encoder_replay_with_eager_decoder_matches_eager(stages):
    with torch.inference_mode():
        # DP's request-count bound can exceed the ladder even though its real
        # narrowed rows fit. On DP1 an empty window takes the eager decoder.
        counts = [20] if stages.dp_size == 1 else [4] * 5
        completes = [False] if stages.dp_size == 1 else [True] * 5
        ctx, ids, positions, kwargs = stages.prepare(
            counts, completes=completes, prefix=0, trial=0
        )
        expected, expected_aux = stages.eager(ctx, ids, positions, kwargs)
        expected = expected.clone()
        expected_aux = [value.clone() for value in expected_aux]
        assert stages.graph.can_run(ctx)
        actual, auxiliary = stages.graph.replay(
            ctx, ids, positions=positions, model_kwargs=kwargs
        )
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        for actual_tap, expected_tap in zip(auxiliary, expected_aux, strict=True):
            torch.testing.assert_close(actual_tap, expected_tap, atol=1e-5, rtol=1e-5)
        assert ctx.input_num_tokens == 20


@pytest.mark.parametrize("dp_size", [1, 4])
def test_multimodal_mask_refreshes_across_independent_graph_stages(dp_size):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    with torch.inference_mode():
        owner = StageHarness(
            dp_size,
            encoder_sizes=[128, 256, 384],
            decoder_sizes=[128, 256, 384],
            model=MaskedCausalLM(),
        )
        # Cover each small bucket, encoder-only, decoder-only, full eager,
        # padding after images, and a rank with no local decoder rows.
        cases = (
            ([121], True, False),
            ([211], True, True),
            ([333], True, True),
            ([119], True, False),
            ([451], True, True),
            ([61] * 4, True, True),
            ([111] * 4, True, True),
            ([111], False, True),
            ([117], True, False),
        )
        for trial, (counts, complete, images) in enumerate(cases):
            ctx, ids, positions, kwargs = owner.prepare(
                counts, completes=[complete] * len(counts), prefix=16, trial=trial
            )
            mask = torch.zeros_like(ids, dtype=torch.bool)
            if images:
                mask[1::3] = True
                kwargs["image_mask"] = mask
                kwargs["input_embeds"] = owner.model.model.embed_tokens(ids) + 7
            mm_context = (
                MultimodalForwardContext([], [], [], 512)
                if images or trial == 3
                else None
            )
            live = owner.model.prepare_prefill_graph_inputs(
                ids, ctx, model_kwargs=kwargs, multimodal_context=mm_context
            )
            owner.model.model.encoder_cache.fill_(-991)
            expected, expected_aux = owner.eager(ctx, ids, positions, live)
            expected = expected.clone()
            expected_aux = [value.clone() for value in expected_aux]
            expected_cache = owner.model.model.encoder_cache.clone()
            owner.model.model.encoder_cache.fill_(-991)
            if owner.graph.can_run(ctx, mm_context):
                actual, auxiliary = owner.graph.replay(
                    ctx, ids, mm_context, positions=positions, model_kwargs=kwargs
                )
                encoder_bucket = owner.graph._replay_bucket(ctx)
                if encoder_bucket is not None:
                    staged = owner.graph._model_input_buffers["image_mask"]
                    torch.testing.assert_close(staged[: ids.numel()], mask)
                    assert not staged[ids.numel() : encoder_bucket].any()
                decoder_bucket = owner.graph._decoder_replay_bucket(ctx)
                if decoder_bucket is not None:
                    staged = owner.graph._decoders[decoder_bucket].statics.image_mask
                    kept = owner.backend.kept_rows
                    torch.testing.assert_close(staged[: kept.numel()], mask[kept])
                    assert not staged[kept.numel() :].any()
            else:
                actual, auxiliary = owner.eager(ctx, ids, positions, live)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            for value, reference in zip(auxiliary, expected_aux, strict=True):
                torch.testing.assert_close(value, reference, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(owner.model.model.encoder_cache, expected_cache)
