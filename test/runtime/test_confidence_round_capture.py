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

"""Replay the real executor stages with small replacement model weights."""

import gc
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist
from transformers import PretrainedConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, suite="runtime-1gpu")

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.confidence_graph import (
    ConfidenceDecodeGraphs,
    DecodeStages,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.drafter.base import BaseDrafter
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.execution.forward_step import (
    DeepEPCudaGraphRunnerAdapter,
    ForwardStepRunner,
)
from tokenspeed.runtime.execution.input_buffer import InputBuffers
from tokenspeed.runtime.execution.model_executor import ModelExecutor
from tokenspeed.runtime.execution.nan_guard import NanGuard
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.execution.runtime_states import RuntimeStates
from tokenspeed.runtime.execution.verify_layout import ConfidenceFeedback
from tokenspeed.runtime.grammar.capturable_grammar import (
    CapturableGrammarExecutor,
    EagerGrammarBuffers,
)
from tokenspeed.runtime.layers.attention.backends.base import CachePoolBinding
from tokenspeed.runtime.models.base.causal_lm import BaseCausalLM
from tokenspeed.runtime.models.dspark import AcceptRatePredictor
from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
from tokenspeed.runtime.sampling.backends.greedy import GreedySamplingBackend
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph capture"
)


@pytest.fixture(autouse=True)
def _collect_graph_owners_after_test():
    yield
    # Runner callbacks form cycles. Free their graph pools before a later test
    # starts another capture, where CUDA graph destruction is forbidden.
    torch.cuda.synchronize()
    gc.collect()


@dataclass(frozen=True)
class _Config:
    max_req_pool_size: int = 9
    output_length: int = 4
    spec_num_tokens: int = 4
    spec_draft_tree: bool = False
    pp_size: int = 1
    pp_rank: int = 0
    data_parallel_size: int = 1
    model_is_mrope: bool = False


class _Backend(CachePoolBinding):
    # Transformer weights/cache arithmetic are replaced by a tiny model. The
    # executor's real sampler, stop policy, candidate publication and frontier
    # kernel still run, including their writes through padding slots.
    def __init__(self, buffers):
        self.buffers = buffers
        self.bs = 0
        self.verify_values = torch.zeros(32, 4, device="cuda")
        self.context_kv = torch.zeros(40, 4, device="cuda")
        self.committed = torch.zeros(10, 4, device="cuda")
        self.commit_counts = torch.zeros(10, dtype=torch.int32, device="cuda")

    def refresh_decode_metadata(
        self, bs, actual_bs, req_pool_indices, seq_lens, **kwargs
    ):
        self.bs = bs

    def decode_window_locations(self):
        return (
            self.buffers.req_pool_indices_buf[: self.bs, None] * 4
            + torch.arange(4, device="cuda")
        ).flatten()

    def commit_state_after_acceptance(self, lengths, *, num_extends, accepted_path):
        slots = self.buffers.state_write_req_pool_indices_buf[: lengths.numel()]
        valid = (slots > 0) & (slots < 9)
        rows = torch.arange(4, device="cuda")[None, :] < lengths[:, None]
        values = self.verify_values[: lengths.numel() * 4].view(-1, 4, 4)
        delta = torch.where((rows & valid[:, None])[:, :, None], values, 0).sum(1)
        self.committed.index_add_(0, slots, delta)
        self.commit_counts.index_add_(0, slots, (valid & (lengths > 0)).int())


class _Body(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 4)
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.arange(64).reshape(16, 4).float() / 64 + 0.25
            )

    def forward(self, ids, positions, ctx):
        dense = self.embedding(ids.long()) + positions[:, None] / 100
        ctx.attn_backend.verify_values[: dense.shape[0]].copy_(dense)
        hidden = (
            dense
            if ctx.verify_queries is None
            else ctx.verify_queries.pack(dense, token_dim=0)
        )
        return hidden, [hidden * 2, hidden + 1]


class _LanguageModel(BaseCausalLM):
    def resolve_model(self, config, mapping, quant_config, prefix):
        return _Body()

    def resolve_lm_head(self, config, quant_config, prefix):
        head = torch.nn.Linear(4, 16, bias=False)
        with torch.no_grad():
            head.weight.fill_(-0.5)
            head.weight[3].fill_(1)
        return head


class _ModelRunner:
    def __init__(self, model):
        self.model = model

    def forward(self, ctx, input_ids, positions, **kwargs):
        return self.model(ctx, input_ids, positions, **kwargs)


class _Draft(BaseDrafter):
    def __init__(self, states, buffers, backend):
        super().__init__(
            4,
            runtime_states=states,
            input_buffers=buffers,
            attn_backend=backend,
            vocab_size=16,
        )
        self.verify_policy = "confidence"
        self.draft_query_width = 3
        self.draft_seq_lens_buf = torch.zeros(8, dtype=torch.int32, device="cuda")
        self.attention_kind = "qwen_mha"
        self.projection = torch.nn.Linear(8, 4, bias=False, device="cuda")
        self.confidence = AcceptRatePredictor(4, 0, with_markov=False).cuda()
        with torch.no_grad():
            self.projection.weight.copy_(
                torch.arange(32, device="cuda").reshape(4, 8) / 64
            )
            self.confidence.proj.weight.fill_(0.02)
            self.confidence.proj.bias.fill_(0.1)

    def run(self, base_ctx, logits_output, output_tokens, accept_lengths):
        batch = base_ctx.bs
        slots = self.input_buffers.req_pool_indices_buf[:batch]
        projected = self.projection(logits_output.hidden_states).view(batch, 4, 4)
        self.attn_backend.context_kv.index_copy_(
            0, self.attn_backend.decode_window_locations(), projected.flatten(0, 1)
        )
        self.draft_seq_lens_buf[:batch].copy_(
            self.runtime_states.valid_cache_lengths[slots] + accept_lengths
        )
        row = (accept_lengths - 1).clamp(min=0).long()
        anchor = output_tokens.view(batch, 4).gather(1, row[:, None])
        proposal = (anchor + torch.arange(4, device="cuda")[None, :]) % 16
        rates = self.confidence(projected[:, :3].flatten(0, 1), None).view(batch, 3)
        self.runtime_states.publish_draft_confidence(
            self.input_buffers.state_write_req_pool_indices_buf[:batch],
            rates,
            group=None,
        )
        return proposal


class _Executor(ModelExecutor):
    def __init__(self, *, logprobs, grammar):
        # Replacement model weights keep this integration test small. All
        # target/draft boundary and state-update methods are the real executor's.
        self.config = _Config()
        self.device = "cuda"
        self.device_module = torch.cuda
        self.input_buffers = InputBuffers(8, 32, 9, device="cuda")
        self.runtime_states = RuntimeStates(9, 16, 4, "cuda")
        self.runtime_states.future_input_map.zero_()
        self.runtime_states.init_draft_confidence(3)
        self.runtime_states.draft_confidence.fill_(0.8)
        self.attn_backend = _Backend(self.input_buffers)
        self.drafter = _Draft(
            self.runtime_states, self.input_buffers, self.attn_backend
        )
        self.model_runner = _ModelRunner(
            _LanguageModel(
                PretrainedConfig(hidden_size=4, vocab_size=16), Mapping(rank=0)
            ).cuda()
        )
        self.model_runner.model.logits_processor.prepare_communication(
            self.model_runner.model.lm_head
        )
        if not logprobs and not grammar:
            self.model_runner.model.logits_processor.fuse_argmax()
        self.sampling_backend = GreedySamplingBackend(
            SamplingBackendConfig(
                synthetic_acceptance_length=None,
                enable_speculative_sampling=False,
                sampling_stream="batch",
                logprob_order="torch",
                max_bs=8,
                max_draft_tokens_per_req=4,
                max_req_pool_size=10,
                vocab_size=16,
                device="cuda",
                enable_output_logprobs=logprobs,
                enable_output_cumulative_probs=logprobs,
            )
        )
        self.nan_guard = NanGuard(8, "cuda")
        self.tree_spec = None
        self.grammar_runtime = None
        self.dspark_context_producer = None
        self._active_positions_override = None
        self._active_multimodal_context = None
        self._simulated_accept_length = None
        self._draft_final_step_counter = None


class _Runner(ForwardStepRunner):
    def __init__(self, executor, *, grammar, budget_policy):
        self.config = executor.config
        self.max_tokens_per_req = 4
        self.capture_bs = [1, 2, 4]
        self.max_capture_bs = 4
        self.max_decode_bs = 8
        self.device = "cuda"
        self.vocab_size = 16
        self.drafter = executor.drafter
        self.attn_backend = executor.attn_backend
        self.runtime_states = executor.runtime_states
        self.input_buffers = executor.input_buffers
        self.token_to_kv_pool = None
        self.capturable_grammar = None
        self.eager_grammar_buffers = (
            EagerGrammarBuffers(8, 16, 4, "cuda") if grammar else None
        )
        self.grammar_backend = "xgrammar" if grammar else "none"
        self.verify_query_layouts = {}
        self.stream = torch.cuda.Stream()
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(enable=False)
        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()
        self._graph_debug = True
        self.enable_cudagraph_gc = False
        self.confidence_feedback = ConfidenceFeedback(
            4, 2, 8, budget_policy=budget_policy
        )
        self.update_runtime_state = executor._update_forward_runtime_state
        self.decode_stages = DecodeStages(
            executor._forward_target,
            executor._forward_draft,
            executor._forward_outputs,
            executor._confidence_graph_inputs_supported,
        )
        self.segmented_decode = ConfidenceDecodeGraphs(self, self.decode_stages)

    def placeholder_block_tables(self, bs):
        return {}

    def _init_capture_metadata(self, bs):
        self.attn_backend.refresh_decode_metadata(
            bs,
            0,
            self.input_buffers.req_pool_indices_buf[:bs],
            self.input_buffers.seq_lens_buf[:bs],
        )

    def warmup_decode_path(self, batch_sizes, *, graph_phase):
        # The stage owner does four ordinary warmups per captured bucket.
        pass


class _Observer:
    def measure(self, series):
        return nullcontext()


@pytest.fixture(scope="module")
def process_group(tmp_path_factory):
    owns_group = not dist.is_initialized()
    if owns_group:
        location = tmp_path_factory.mktemp("confidence-round") / "group"
        dist.init_process_group(
            "gloo", init_method=f"file://{location}", rank=0, world_size=1
        )
    yield
    if owns_group:
        dist.destroy_process_group()


def _inputs(executor, runner, slots, *, queries, step, graph):
    buffers = executor.input_buffers
    batch = len(slots)
    buffers.req_pool_indices_buf[:batch] = torch.tensor(slots, device="cuda")
    buffers.state_write_req_pool_indices_buf[:batch] = torch.tensor(
        slots, device="cuda"
    )
    buffers.input_ids_buf[: batch * 4].fill_(3)
    if step % 2:
        buffers.input_ids_buf[: batch * 4].view(batch, 4)[:, 2] = 4
    buffers.positions_buf[: batch * 4] = (
        executor.runtime_states.valid_cache_lengths[
            buffers.req_pool_indices_buf[:batch], None
        ]
        .long()
        .add(torch.arange(4, device="cuda"))
        .flatten()
    )
    buffers.seq_lens_buf[:batch] = (
        executor.runtime_states.valid_cache_lengths[
            buffers.req_pool_indices_buf[:batch]
        ]
        + 4
    )
    buffers.input_lengths_buf[:batch].fill_(4)
    buffers.force_single_token_verify_buf.zero_()
    capacity = min(4 if graph else 8, queries)
    ctx = ForwardContext(
        attn_backend=executor.attn_backend,
        token_to_kv_pool=None,
        bs=batch,
        num_extends=0,
        input_num_tokens=batch * 4,
        forward_mode=ForwardMode.DECODE,
        output_layout=ForwardOutputLayout(0, 0, batch, 4),
        capture_hidden_mode=CaptureHiddenMode.FULL,
        verify_queries=runner.query_layout(capacity, queries),
        all_decode_or_idle=True,
    )
    executor.nan_guard.reset(batch)
    runner.segmented_decode.bind_round(
        ctx, actual_bs=batch, block_tables={}, use_graph=graph
    )
    info = runner.segmented_decode._sampling_info(batch)
    return ctx, info


@pytest.mark.parametrize("graph", [False, True])
def test_confidence_feedback_keeps_acceptance_before_token_limit(process_group, graph):
    executor = _Executor(logprobs=False, grammar=False)
    runner = _Runner(executor, grammar=False, budget_policy="confidence")
    if graph:
        runner.capture(entries=None, observer=_Observer())
    states = executor.runtime_states
    states.spec_candidates_ready[1] = True
    # The uncached anchor already consumes one position of the token limit.
    states.token_limits[1] = states.valid_cache_lengths[1] + 2
    ctx, info = _inputs(executor, runner, [1], queries=4, step=0, graph=graph)
    result = runner.segmented_decode.run_round(ctx, info)
    assert result[1].tolist() == [1]
    assert ctx.verify_queries.accept_lengths[:1].tolist() == [4]


@pytest.mark.parametrize(
    "logprobs,grammar", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("budget_policy", ["confidence", "history"])
def test_captured_round_preserves_outputs_and_state_when_b_and_q_change(
    process_group, logprobs, grammar, budget_policy
):
    eager, captured = (_Executor(logprobs=logprobs, grammar=grammar) for _ in range(2))
    reference, runner = (
        _Runner(executor, grammar=grammar, budget_policy=budget_policy)
        for executor in (eager, captured)
    )
    runner.capture(entries=None, observer=_Observer())
    cases = [
        (8, [5, 2, 7, 1]),
        (8, [2]),
        (4, [7, 5, 2]),
        (12, [5, 1]),
        (8, [7, 1, 5, 2]),
        (1, [1]),
    ]
    for step, (queries, slots) in enumerate(cases):
        for executor in (eager, captured):
            executor.runtime_states.spec_candidates_ready[1:9] = step != 0
            if step == 3:
                executor.runtime_states.token_limits[5] = (
                    executor.runtime_states.valid_cache_lengths[5] + 2
                )
        actual_ctx, actual_info = _inputs(
            captured, runner, slots, queries=queries, step=step, graph=True
        )
        expected_ctx, expected_info = _inputs(
            eager, reference, slots, queries=queries, step=step, graph=False
        )
        if grammar:
            for owner in (runner, reference):
                owner.eager_grammar_buffers.bitmask.fill_(-1)
                owner.eager_grammar_buffers.bitmask[1::4] = 1 << (4 if step % 2 else 3)
            if step == 3:
                # Same base address, different row stride: replay must honor
                # this explicit mask instead of using the capture's view.
                for owner in (runner, reference):
                    owner.eager_grammar_buffers.bitmask[8] = 1 << 4
                actual_info.vocab_mask = runner.eager_grammar_buffers.bitmask[::2][
                    : len(slots) * 4
                ]
                expected_info.vocab_mask = reference.eager_grammar_buffers.bitmask[::2][
                    : len(slots) * 4
                ]
        expected = reference.segmented_decode.run_round(expected_ctx, expected_info)
        actual = runner.segmented_decode.run_round(actual_ctx, actual_info)
        for got, want in zip(actual[:4], expected[:4]):
            if got is not None:
                torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-6)
        for got, want in (
            (
                captured.runtime_states.valid_cache_lengths,
                eager.runtime_states.valid_cache_lengths,
            ),
            (
                captured.runtime_states.future_input_map[1:9],
                eager.runtime_states.future_input_map[1:9],
            ),
            (
                captured.runtime_states.draft_confidence[1:9],
                eager.runtime_states.draft_confidence[1:9],
            ),
            (
                captured.runtime_states.previous_draft_confidence[1:9],
                eager.runtime_states.previous_draft_confidence[1:9],
            ),
            (
                captured.runtime_states.draft_confidence_generation[1:9],
                eager.runtime_states.draft_confidence_generation[1:9],
            ),
            (captured.runtime_states.stopped, eager.runtime_states.stopped),
            (captured.attn_backend.committed, eager.attn_backend.committed),
            (captured.attn_backend.commit_counts, eager.attn_backend.commit_counts),
            (
                captured.attn_backend.context_kv[4:36],
                eager.attn_backend.context_kv[4:36],
            ),
        ):
            torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-6)
        torch.testing.assert_close(
            captured.nan_guard.flags[: len(slots)], eager.nan_guard.flags[: len(slots)]
        )


def test_explicit_decode_override_and_above_ladder_keep_the_same_stage_behavior(
    process_group,
):
    executor = _Executor(logprobs=False, grammar=False)
    runner = _Runner(executor, grammar=False, budget_policy="confidence")
    runner.capture(entries=None, observer=_Observer())
    ctx, info = _inputs(executor, runner, [1, 2], queries=8, step=2, graph=True)
    ctx.decode_input_ids = [3, 3]
    executor.input_buffers.force_single_token_verify_buf[:2].fill_(True)
    executor.runtime_states.spec_candidates_ready[1:3].fill_(True)
    result = runner.segmented_decode.run_round(ctx, info)
    torch.testing.assert_close(
        result[1], torch.ones(2, dtype=torch.int32, device="cuda")
    )
    for persistent in (True, False):
        ctx, info = _inputs(executor, runner, [1, 2], queries=8, step=2, graph=True)
        positions = executor.input_buffers.positions_buf[:8]
        executor._active_positions_override = (
            positions if persistent else positions.clone() + 200
        )
        assert executor._confidence_graph_inputs_supported(ctx) == persistent
        before = executor.attn_backend.committed[1:3].clone()
        hidden = executor.model_runner.model.model.embedding(
            executor.input_buffers.input_ids_buf[:8].long()
        )
        expected = before + (
            hidden + executor._active_positions_override[:, None] / 100
        ).view(2, 4, 4).sum(1)
        result = runner.segmented_decode.run_round(ctx, info)
        torch.testing.assert_close(
            result[1], torch.full((2,), 4, dtype=torch.int32, device="cuda")
        )
        torch.testing.assert_close(
            executor.attn_backend.committed[1:3], expected, atol=1e-5, rtol=1e-6
        )
    executor._active_positions_override = None
    ctx, info = _inputs(
        executor, runner, [1, 2, 3, 4, 5, 6], queries=12, step=2, graph=False
    )
    before = executor.runtime_states.valid_cache_lengths.clone()
    result = runner.segmented_decode.run_round(ctx, info)
    torch.testing.assert_close(
        executor.runtime_states.valid_cache_lengths[1:7] - before[1:7], result[1]
    )
    assert result[0].shape == (24,)


def test_captured_grammar_host_node_uses_the_live_request_batch(process_group):
    xgrammar = pytest.importorskip("xgrammar")
    from test.runtime.grammar_test_utils import host_grammar

    vocab = [f"t{index}" for index in range(16)]
    vocab[0], vocab[3], vocab[4] = "<eos>", "a", "b"
    compiled = xgrammar.GrammarCompiler(
        xgrammar.TokenizerInfo(vocab, stop_token_ids=[0])
    ).compile_grammar('root ::= "b"{128}')
    grammars = {slot: host_grammar(compiled) for slot in (1, 2, 3)}
    executor = _Executor(logprobs=False, grammar=True)
    runner = _Runner(executor, grammar=False, budget_policy="confidence")
    runtime = CapturableGrammarExecutor(8, 16, 4)
    executor.grammar_runtime = runtime
    runner.capturable_grammar = runtime
    runner.grammar_backend = "xgrammar"
    runner.capture(entries=None, observer=_Observer())
    executor.runtime_states.spec_candidates_ready[1:9].fill_(True)
    for step, (queries, slots) in enumerate([(8, [1, 2]), (4, [2]), (8, [2, 1, 3])]):
        ctx, info = _inputs(
            executor, runner, slots, queries=queries, step=step, graph=True
        )
        completion = runtime.add_batch(
            grammars=[grammars[slot] for slot in slots],
            bs=len(slots),
            has_candidates=True,
            output_layout=ctx.output_layout,
        )
        tokens, lengths, _, _, _ = runner.segmented_decode.run_round(ctx, info)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            lengths, torch.ones(len(slots), dtype=torch.int32, device="cuda")
        )
        torch.testing.assert_close(
            tokens.view(-1, 4)[:, 0],
            torch.full((len(slots),), 4, dtype=torch.int32, device="cuda"),
        )
        completion.advance_on_host(tokens.cpu().tolist(), lengths.cpu().tolist())
        assert completion.terminated_at() == [-1] * len(slots)
