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

"""Q-keyed target/verification graphs and B-keyed draft/commit graphs.

Both graphs call the ordinary executor stages. Address-stable handoff buffers
keep the request graph independent of the target query bucket.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch
import torch.distributed as dist
from tokenspeed_kernel.workspace import capture_session

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.cudagraph_memory import (
    CapturedLadder,
    probe_positions,
)
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.execution.graph_ptr_guard import (
    snapshot_graph_metadata,
    verify_graph_metadata,
)
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.grammar.capturable_grammar import bind_grammar_mask_buf
from tokenspeed.runtime.layers.logits_processor_output import LogitsProcessorOutput
from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.nvtx import nvtx_range

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.forward_step import ForwardStepRunner
    from tokenspeed.runtime.execution.memory_delta import MemoryDeltaObserver

logger = get_colorful_logger(__name__)


@dataclass(frozen=True)
class DecodeStages:
    """The same executor operations used by eager and captured forwards."""

    target: Callable
    draft: Callable
    outputs: Callable
    inputs_supported: Callable[[ForwardContext], bool]


@dataclass
class DecodeStageGraph:
    graph: torch.cuda.CUDAGraph
    output: tuple
    milliseconds: float
    metadata: dict[str, dict[str, tuple]] | None
    sampling_info: SamplingBatchInfo | None


class DecodeStageBuffers:
    """Batch-row handoff, never slot-owned or vocabulary-sized storage.

    Each Q graph writes its fixed request extent. A B graph can read any
    prefix within that extent without binding the Q graph's allocations.
    """

    def __init__(self, max_bs: int, width: int, device: torch.device):
        self.width = width
        self.tokens = torch.zeros(max_bs * width, dtype=torch.int32, device=device)
        self.lengths = torch.zeros(max_bs, dtype=torch.int32, device=device)
        self.hidden_states: torch.Tensor | None = None
        self.empty_logits = torch.empty((0, 0), device=device)

    def publish(self, output: LogitsProcessorOutput, tokens, lengths) -> None:
        hidden = output.hidden_states
        if hidden is None:
            raise ValueError("confidence drafting requires target hidden states")
        if self.hidden_states is None:
            if hidden.is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "decode stage buffers must be initialized before capture"
                )
            self.hidden_states = hidden.new_zeros(
                (self.tokens.numel(), hidden.shape[1])
            )
        if (
            hidden.shape[0] != tokens.numel()
            or tokens.numel() != lengths.numel() * self.width
        ):
            raise ValueError("decode handoff rows must match request windows")
        if (
            hidden.shape[1:] != self.hidden_states.shape[1:]
            or hidden.dtype != self.hidden_states.dtype
        ):
            raise ValueError(
                "target capture features cannot change between query buckets"
            )
        rows = tokens.numel()
        self.hidden_states[:rows].copy_(hidden)
        self.tokens[:rows].copy_(tokens)
        self.lengths[: lengths.numel()].copy_(lengths)

    def draft_output(self, batch: int) -> LogitsProcessorOutput:
        if self.hidden_states is None:
            raise RuntimeError("target stage must initialize the decode handoff")
        return LogitsProcessorOutput(
            next_token_logits=self.empty_logits,
            hidden_states=self.hidden_states[: batch * self.width],
        )


class ConfidenceDecodeGraphs:
    """Capture target + verify by Q, and draft + accepted-state commit by B.

    The Q stage's request capacity is ``min(max_capture_bs, Q)``. Device slot
    masks make its unused requests empty, so B is not another target key.
    """

    def __init__(self, runner: ForwardStepRunner, stages: DecodeStages):
        self.runner = runner
        self.stages = stages
        self.drafter = runner.drafter
        self.width = runner.max_tokens_per_req
        self.target_sizes = sorted(
            {b * k for b in runner.capture_bs for k in range(1, self.width + 1)}
        )
        self.targets: dict[int, DecodeStageGraph] = {}
        self.drafts: dict[int, DecodeStageGraph] = {}
        self.pool = None
        self.use_graph = False
        self.actual_bs = 0
        self.block_tables: dict | None = None
        self.buffers = DecodeStageBuffers(
            runner.max_capture_bs, self.width, runner.input_buffers.input_ids_buf.device
        )

    @property
    def capture_plan(self) -> dict[str, list[int]]:
        return {
            "target": list(reversed(self.target_sizes)),
            "drafter": sorted(self.runner.capture_bs, reverse=True),
        }

    def capture_ladders(self, entries: int | None) -> dict[str, CapturedLadder]:
        return {
            f"decode:{name}": CapturedLadder(
                sizes, probe_positions(len(sizes), entries)
            )
            for name, sizes in self.capture_plan.items()
        }

    def bind_round(
        self,
        ctx: ForwardContext,
        *,
        actual_bs: int,
        block_tables: dict | None,
        use_graph: bool,
    ) -> None:
        self.actual_bs = actual_bs
        self.block_tables = block_tables
        self.use_graph = use_graph
        capacity = ctx.verify_queries.batch_size
        ib = self.runner.input_buffers
        sentinel = int(self.runner.config.max_req_pool_size)
        if capacity > actual_bs:
            ib.req_pool_indices_buf[actual_bs:capacity].fill_(sentinel)
            ib.state_write_req_pool_indices_buf[actual_bs:capacity].fill_(sentinel)
            ib.seq_lens_buf[actual_bs:capacity].fill_(1)
            ib.input_lengths_buf[actual_bs:capacity].fill_(self.width)
            ib.force_single_token_verify_buf[actual_bs:capacity].zero_()

    def _refresh_target(self, capacity: int, *, capture: bool) -> None:
        runner = self.runner
        if capture:
            runner._init_capture_metadata(capacity)
            return
        tables = self.block_tables
        if self.actual_bs == 0 and not tables:
            tables = runner.placeholder_block_tables(capacity)
        runner.attn_backend.refresh_decode_metadata(
            capacity,
            self.actual_bs,
            runner.input_buffers.req_pool_indices_buf[:capacity],
            runner.input_buffers.seq_lens_buf[:capacity],
            forward_mode=ForwardMode.DECODE,
            block_tables=tables,
            for_graph_replay=self.use_graph,
            num_tokens=capacity * self.width,
        )

    def _refresh_draft(self, bs: int) -> None:
        draft = self.drafter
        draft.attn_backend.refresh_decode_metadata(
            bs,
            self.actual_bs,
            draft.input_buffers.req_pool_indices_buf[:bs],
            draft.draft_seq_lens_buf[:bs] + draft.draft_query_width,
            forward_mode=ForwardMode.DECODE,
            block_tables=draft.round_block_tables,
            num_extends=0 if draft.attention_kind == "kimi_mla" else bs,
        )

    def _context(self, capacity: int, queries: int) -> ForwardContext:
        return ForwardContext(
            attn_backend=self.runner.attn_backend,
            token_to_kv_pool=self.runner.token_to_kv_pool,
            bs=capacity,
            num_extends=0,
            input_num_tokens=capacity * self.width,
            forward_mode=ForwardMode.DECODE,
            output_layout=ForwardOutputLayout(0, 0, capacity, self.width),
            capture_hidden_mode=CaptureHiddenMode.FULL,
            all_decode_or_idle=True,
            verify_queries=self.runner.query_layout(capacity, queries),
        )

    def _sampling_info(self, capacity: int) -> SamplingBatchInfo:
        runner = self.runner
        states = runner.runtime_states
        info = SamplingBatchInfo(
            req_pool_indices=runner.input_buffers.req_pool_indices_buf[:capacity],
            valid_cache_lengths=states.valid_cache_lengths,
            draft_probs=states.draft_probs,
            vocab_size=runner.vocab_size,
            device=runner.device,
        )
        bind_grammar_mask_buf(
            info,
            runner.eager_grammar_buffers,
            capacity,
            spec=True,
            capturable=runner.capturable_grammar,
            grammar_backend=runner.grammar_backend,
        )
        return info

    @staticmethod
    def _same_sampling_inputs(
        live: SamplingBatchInfo, captured: SamplingBatchInfo
    ) -> bool:
        # Explicit batch scalars are not persistent pool parameters. Respect
        # them in the ordinary stage rather than replaying an omitted input.
        if (
            any(
                value is not None
                for value in (
                    live.temperatures,
                    live.top_ps,
                    live.top_ks,
                    live.min_ps,
                    live.extend_lengths,
                )
            )
            or live.batch_row_offset
        ):
            return False
        if live.vocab_size != captured.vocab_size or live.device != captured.device:
            return False
        for actual, expected, request_rows in (
            (live.req_pool_indices, captured.req_pool_indices, True),
            (live.valid_cache_lengths, captured.valid_cache_lengths, False),
            (live.draft_probs, captured.draft_probs, False),
            (live.vocab_mask, captured.vocab_mask, True),
        ):
            if actual is None or expected is None:
                if actual is not expected:
                    return False
            elif (
                actual.data_ptr() != expected.data_ptr()
                or actual.dtype != expected.dtype
                or actual.device != expected.device
                or actual.stride() != expected.stride()
                or actual.shape[1:] != expected.shape[1:]
                or (actual.shape != expected.shape and not request_rows)
            ):
                return False
        return live.apply_vocab_mask is captured.apply_vocab_mask

    def _replay(self, entry: DecodeStageGraph, *, name: str, backends: dict) -> None:
        self.runner.deepep_adapter.replay()
        if entry.metadata is not None:
            for key, metadata in entry.metadata.items():
                verify_graph_metadata(backends[key], metadata, context=f"{name}, {key}")
        with nvtx_range(name, color="red"):
            entry.graph.replay()

    def _publish_target(self, ctx: ForwardContext, info: SamplingBatchInfo) -> tuple:
        output, tokens, lengths = self.stages.target(ctx.bs, ctx, info)
        self.buffers.publish(output, tokens, lengths)
        return self.stages.outputs(
            output,
            self.buffers.tokens[: tokens.numel()],
            self.buffers.lengths[: ctx.bs],
        )

    def _finish_draft(self, ctx: ForwardContext) -> tuple:
        tokens = self.buffers.tokens[: ctx.bs * self.width]
        lengths = self.buffers.lengths[: ctx.bs]
        self.stages.draft(ctx, self.buffers.draft_output(ctx.bs), tokens, lengths)
        result = (tokens, lengths, None, None, None)
        self.runner.commit_forward(ctx, result)
        return result

    def _prepare_draft_capture(self, capacity: int) -> None:
        self._refresh_target(capacity, capture=True)
        self.runner.attn_backend.prepare_state_commit_capture()

    def run_round(self, ctx: ForwardContext, sampling_info: SamplingBatchInfo) -> tuple:
        from tokenspeed.runtime.execution.forward_step import live_graph_outputs

        layout = ctx.verify_queries
        capacity = layout.batch_size
        batch = (
            self.runner.capture_bs[bisect_left(self.runner.capture_bs, ctx.bs)]
            if self.use_graph
            else ctx.bs
        )
        target = self.targets.get(layout.query_capacity) if self.use_graph else None
        draft = self.drafts.get(batch) if self.use_graph else None
        replay = (
            target is not None
            and draft is not None
            and target.sampling_info is not None
            and self.stages.inputs_supported(ctx)
            and self._same_sampling_inputs(sampling_info, target.sampling_info)
        )
        self._refresh_target(capacity, capture=False)
        if replay:
            self._replay(
                target,
                name=f"target_verify_graph Q={layout.query_capacity}",
                backends={"target": self.runner.attn_backend},
            )
            self._refresh_target(batch, capture=False)
            self._refresh_draft(batch)
            self._replay(
                draft,
                name=f"draft_commit_graph B={batch}",
                backends={
                    "target": self.runner.attn_backend,
                    "draft": self.drafter.attn_backend,
                },
            )
            result = target.output
        else:
            # The ordinary stage keeps every live input and the actual request
            # extent. The backend still serves the same compact query layout.
            output, tokens, lengths = self.stages.target(ctx.bs, ctx, sampling_info)
            self._refresh_target(ctx.bs, capture=False)
            self.stages.draft(ctx, output, tokens, lengths)
            result = self.stages.outputs(output, tokens, lengths)
            self.runner.commit_forward(ctx, result)
        return live_graph_outputs(result, ctx.bs, self.width)

    def _queue_grammar(self, ctx: ForwardContext) -> None:
        grammar = self.runner.capturable_grammar
        if grammar is not None:
            grammar.add_batch(
                grammars=[None] * ctx.bs,
                bs=ctx.bs,
                has_candidates=False,
                output_layout=ctx.output_layout,
            )

    def _capture(
        self,
        run: Callable,
        prepare: Callable,
        before_run: Callable,
        backends: dict,
        sampling_info: SamplingBatchInfo | None,
    ) -> DecodeStageGraph:
        from tokenspeed.runtime.execution.forward_step import decode_graph_capture

        runner = self.runner
        with torch.cuda.stream(runner.stream):
            for _ in range(4):
                prepare()
                before_run()
                run()
            torch.cuda.synchronize()
            dist.barrier()
            prepare()
            before_run()
            runner.deepep_adapter.capture()
            graph = torch.cuda.CUDAGraph()
            with decode_graph_capture(), capture_session(
                graph
            ), runner.memory_saver_adapter.cuda_graph(
                graph, pool=self.pool, stream=runner.stream, tag="cuda_graph"
            ):
                output = run()
            self.pool = graph.pool()
            if sampling_info is not None and runner.capturable_grammar is not None:
                # Capture records the host node but does not consume its batch.
                runner.capturable_grammar.reset_state()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            start.record()
            for _ in range(5):
                before_run()
                graph.replay()
            end.record()
            end.synchronize()
            milliseconds = start.elapsed_time(end) / 5
            if sampling_info is not None and runner.capturable_grammar is not None:
                runner.capturable_grammar.reset_state()
        torch.cuda.current_stream().wait_stream(runner.stream)
        metadata = (
            {
                name: snapshot_graph_metadata(backend)
                for name, backend in backends.items()
            }
            if runner._graph_debug
            else None
        )
        return DecodeStageGraph(graph, output, milliseconds, metadata, sampling_info)

    def capture(self, *, entries: int | None, observer: MemoryDeltaObserver) -> None:
        runner = self.runner
        logger.info(f"Capturing confidence stages: {self.capture_plan}")
        ladders = self.capture_ladders(entries)
        draft_sizes = self.capture_plan["drafter"]
        runner.warmup_decode_path(
            tuple(draft_sizes[index] for index in ladders["decode:drafter"].sampled),
            graph_phase=True,
        )
        for name, sizes in self.capture_plan.items():
            for index in ladders[f"decode:{name}"].sampled:
                size = sizes[index]
                capacity = (
                    min(runner.max_capture_bs, size) if name == "target" else size
                )
                runner.input_buffers.fill_dummy_decode_buffers(
                    batch_size=capacity, total_tokens=capacity * self.width
                )
                runner.input_buffers.seq_lens_buf[:capacity].fill_(self.width)
                ctx = self._context(
                    capacity, size if name == "target" else capacity * self.width
                )
                if name == "target":
                    info = self._sampling_info(capacity)
                    run = lambda ctx=ctx, info=info: self._publish_target(ctx, info)
                    prepare = lambda capacity=capacity: self._refresh_target(
                        capacity, capture=True
                    )
                    before_run = lambda ctx=ctx: self._queue_grammar(ctx)
                    backends = {"target": runner.attn_backend}
                else:
                    info = None
                    run = lambda ctx=ctx: self._finish_draft(ctx)
                    prepare = lambda capacity=capacity: self._prepare_draft_capture(
                        capacity
                    )
                    before_run = lambda: None
                    backends = {
                        "target": runner.attn_backend,
                        "draft": self.drafter.attn_backend,
                    }
                with observer.measure(f"decode:{name}"):
                    entry = self._capture(run, prepare, before_run, backends, info)
                (self.targets if name == "target" else self.drafts)[size] = entry
        # The startup curve composes both captured stages. Completed whole-round
        # feedback additionally includes metadata refresh and graph submission.
        for batch, draft in self.drafts.items():
            for queries, target in self.targets.items():
                if batch <= queries <= batch * self.width:
                    runner.confidence_feedback.planner.set_baseline(
                        batch, queries, target.milliseconds + draft.milliseconds
                    )

    def release(self) -> None:
        self.targets.clear()
        self.drafts.clear()
        self.pool = None
