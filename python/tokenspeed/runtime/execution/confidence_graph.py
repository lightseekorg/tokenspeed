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

"""Independent target-token and draft-request graphs for confidence decode.

The ordinary forward calls the same target body, logits exit, verification and
native draft in order. This owner substitutes only the two captured stages.
Request metadata is refreshed through the backend's existing decode contract.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, replace
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
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.nvtx import nvtx_range

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.forward_step import ForwardStepRunner
    from tokenspeed.runtime.execution.memory_delta import MemoryDeltaObserver
    from tokenspeed.runtime.models.base import BaseCausalLM

logger = get_colorful_logger(__name__)


@dataclass
class DecodeStageGraph:
    graph: torch.cuda.CUDAGraph
    output: object
    milliseconds: float
    metadata: dict[str, tuple] | None


class ConfidenceDecodeGraphs:
    """Target graphs keyed by Q and native draft graphs keyed by B.

    ``min(max_capture_bs, Q)`` is the target graph's fixed request capacity.
    Empty requests carry zero query lengths; real request count stays on the
    outer forward. This capacity is derived from Q, never part of the key.
    """

    def __init__(self, runner: ForwardStepRunner, target_model: BaseCausalLM):
        self.runner = runner
        self.target_model = target_model
        self.drafter = runner.drafter
        self.width = runner.max_tokens_per_req
        self.target_sizes = sorted(
            {b * k for b in runner.capture_bs for k in range(1, self.width + 1)}
        )
        self.targets: dict[int, DecodeStageGraph] = {}
        self.drafts: dict[int, DecodeStageGraph] = {}
        self.pool = None
        self.active = False
        self.use_graph = False
        self.actual_bs = 0
        self.request_capacity = 0
        self.block_tables: dict | None = None
        rows = runner.max_decode_bs * self.width
        ib = runner.input_buffers
        self.input_ids = torch.zeros_like(ib.input_ids_buf[:rows])
        self.positions = torch.zeros_like(
            ib.mrope_positions_buf[:, :rows]
            if runner.config.model_is_mrope
            else ib.positions_buf[:rows]
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
        request_capacity: int,
        block_tables: dict | None,
        use_graph: bool,
    ) -> None:
        self.active = True
        self.actual_bs = actual_bs
        self.request_capacity = request_capacity
        self.block_tables = block_tables
        self.use_graph = use_graph
        ctx.decode_graphs = self
        capacity = ctx.verify_queries.batch_size
        ib = self.runner.input_buffers
        sentinel = int(self.runner.config.max_req_pool_size)
        if capacity > actual_bs:
            ib.req_pool_indices_buf[actual_bs:capacity].fill_(sentinel)
            ib.state_write_req_pool_indices_buf[actual_bs:capacity].fill_(sentinel)
            ib.seq_lens_buf[actual_bs:capacity].fill_(1)

    def _refresh_target(self, capacity: int, actual_bs: int, *, capture: bool) -> None:
        runner = self.runner
        if capture:
            runner._init_capture_metadata(capacity)
            return
        tables = self.block_tables
        if actual_bs == 0 and not tables:
            tables = runner.placeholder_block_tables(capacity)
        runner.attn_backend.refresh_decode_metadata(
            capacity,
            actual_bs,
            runner.input_buffers.req_pool_indices_buf[:capacity],
            runner.input_buffers.seq_lens_buf[:capacity],
            forward_mode=ForwardMode.DECODE,
            block_tables=tables,
            for_graph_replay=self.use_graph,
            num_tokens=capacity * self.width,
        )

    def run_target(self, input_ids, positions, ctx, model_kwargs):
        layout = ctx.verify_queries
        capacity = layout.batch_size
        target_ctx = replace(
            ctx,
            bs=capacity,
            input_num_tokens=capacity * self.width,
            output_layout=ForwardOutputLayout(0, 0, capacity, self.width),
            decode_graphs=None,
            target_capture_sink=None,
        )
        self._refresh_target(capacity, self.actual_bs, capture=False)
        try:
            entry = self.targets.get(layout.query_capacity) if self.use_graph else None
            # Explicit model inputs such as supplied embeddings remain honored.
            # A graph captured without them cannot silently drop them.
            if entry is None or model_kwargs:
                return self.target_model.model(
                    input_ids, positions, target_ctx, **model_kwargs
                )
            self.input_ids[: input_ids.shape[0]].copy_(input_ids)
            self.positions[..., : positions.shape[-1]].copy_(positions)
            self.runner.deepep_adapter.replay()
            if entry.metadata is not None:
                verify_graph_metadata(
                    self.runner.attn_backend,
                    entry.metadata,
                    context=f"target Q={layout.query_capacity}",
                )
            with nvtx_range(f"target_graph Q={layout.query_capacity}", color="red"):
                entry.graph.replay()
            return entry.output
        finally:
            # Verification/commit and the drafter see the actual request
            # extent. The drafter handles its own stage's bucket padding.
            self._refresh_target(self.request_capacity, self.actual_bs, capture=False)

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

    def run_draft(self, current_tokens: torch.Tensor) -> torch.Tensor:
        bs = current_tokens.shape[0]
        capacity = (
            self.runner.capture_bs[bisect_left(self.runner.capture_bs, bs)]
            if self.use_graph
            else bs
        )
        entry = self.drafts.get(capacity) if self.use_graph else None
        if entry is None:
            return self.drafter._draft_native_forward(current_tokens, None, True)
        if capacity > bs:
            draft = self.drafter
            sentinel = int(self.runner.config.max_req_pool_size)
            draft.input_buffers.req_pool_indices_buf[bs:capacity].fill_(sentinel)
            draft.input_buffers.state_write_req_pool_indices_buf[bs:capacity].fill_(
                sentinel
            )
            draft.draft_seq_lens_buf[bs:capacity].zero_()
            draft.block_ids_buf[bs:capacity].zero_()
            draft.block_positions_buf[bs:capacity].copy_(
                draft.block_offsets.expand(capacity - bs, -1)
            )
        self._refresh_draft(capacity)
        self.runner.deepep_adapter.replay()
        if entry.metadata is not None:
            verify_graph_metadata(
                self.drafter.attn_backend,
                entry.metadata,
                context=f"drafter B={capacity}",
            )
        with nvtx_range(f"draft_graph B={capacity}", color="purple"):
            entry.graph.replay()
        return entry.output[:bs]

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

    def _capture(self, run: Callable, prepare: Callable, backend) -> DecodeStageGraph:
        from tokenspeed.runtime.execution.forward_step import decode_graph_capture

        runner = self.runner
        with torch.cuda.stream(runner.stream):
            for _ in range(4):
                prepare()
                run()
            torch.cuda.synchronize()
            dist.barrier()
            prepare()
            runner.deepep_adapter.capture()
            graph = torch.cuda.CUDAGraph()
            with decode_graph_capture(), capture_session(
                graph
            ), runner.memory_saver_adapter.cuda_graph(
                graph, pool=self.pool, stream=runner.stream, tag="cuda_graph"
            ):
                output = run()
            self.pool = graph.pool()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            start.record()
            for _ in range(5):
                graph.replay()
            end.record()
            end.synchronize()
            milliseconds = start.elapsed_time(end) / 5
        torch.cuda.current_stream().wait_stream(runner.stream)
        metadata = snapshot_graph_metadata(backend) if runner._graph_debug else None
        return DecodeStageGraph(graph, output, milliseconds, metadata)

    def capture(self, *, entries: int | None, observer: MemoryDeltaObserver) -> None:
        runner = self.runner
        logger.info(f"Capturing confidence stages: {self.capture_plan}")
        ladders = self.capture_ladders(entries)
        draft_sizes = self.capture_plan["drafter"]
        # The exit and state updates stay eager. Warm the complete forward so
        # their first use does not compile on a live request after startup.
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
                if name == "target":
                    ctx = self._context(capacity, size)
                    states = runner.runtime_states
                    slots = runner.input_buffers.state_write_req_pool_indices_buf[
                        :capacity
                    ]
                    ctx.verify_queries.refresh(
                        states.draft_confidence[slots],
                        states.spec_candidates_ready[slots],
                        torch.zeros(
                            capacity, dtype=torch.bool, device=self.input_ids.device
                        ),
                    )

                    def run(ctx=ctx, capacity=capacity):
                        return self.target_model.model(
                            self.input_ids[: capacity * self.width],
                            self.positions[..., : capacity * self.width],
                            ctx,
                        )

                    prepare = lambda capacity=capacity: self._refresh_target(
                        capacity, 0, capture=True
                    )
                else:

                    def prepare(capacity=capacity):
                        runner._init_capture_metadata(capacity)
                        self.drafter.draft_seq_lens_buf[:capacity].zero_()
                        self.drafter.block_positions_buf[:capacity].copy_(
                            self.drafter.block_offsets.expand(capacity, -1)
                        )

                    def run(capacity=capacity):
                        return self.drafter._draft_native_forward(
                            self.drafter.block_ids_buf[:capacity, 0], None, True
                        )

                with observer.measure(f"decode:{name}"):
                    entry = self._capture(
                        run,
                        prepare,
                        (
                            runner.attn_backend
                            if name == "target"
                            else self.drafter.attn_backend
                        ),
                    )
                (self.targets if name == "target" else self.drafts)[size] = entry
        # The startup curve composes independently measured stages. Completed
        # whole-round observations replace it online, including the eager exit.
        for batch, draft in self.drafts.items():
            for multiple in range(1, self.width + 1):
                queries = batch * multiple
                target = self.targets.get(queries)
                if target is not None:
                    runner.confidence_feedback.planner.observe(
                        batch, queries, target.milliseconds + draft.milliseconds
                    )

    def release(self) -> None:
        self.targets.clear()
        self.drafts.clear()
        self.pool = None
        self.active = False
