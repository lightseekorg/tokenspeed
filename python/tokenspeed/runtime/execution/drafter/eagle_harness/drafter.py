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

"""EAGLE-4 drafters: ``Eagle4`` drafts static chains and trees from the
candidate head, and ``EagleHarness`` EAGLE-4's cost-driven dynamic draft
trees on the EAGLE drafter's lanes.

A round grows a node pool one depth at a time: the root (last verified
token), its ``K`` children from the draft's first step, then at every depth
the frontier's ``K`` best nodes forward as the lane rows and each appends its
``K`` best children. The policy (``DevicePolicy``, also for a one-child
chain) decides after each depth whether to stop, from the measured
cost data and the score deltas the kernels write; the round then verifies the
best nodes in ancestor-first score order, as many as the policy sizes the
verify for.

With ``node_top_k`` lanes the frontier's width is constant, so EAGLE-4's
forwarded-node order is the lanes' slot order: depth ``s`` row
``r`` writes lane slot ``(s - 1) * K + r``.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.sampling.triton import logprob_topk
from tokenspeed_kernel.thirdparty.cuda.eagle_harness import (
    eagle_harness_draft_depth_update,
    eagle_harness_publish_draft_q,
    eagle_harness_sample_children,
    eagle_harness_stable_topk_desc,
    eagle_harness_update_node_data,
    eagle_harness_work_deltas,
    eagle_harness_write_verify_tree,
)
from typing_extensions import override

from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.drafter.eagle import (
    AcceptedPrefixPublisher,
    Eagle,
    EagleDraftInput,
)
from tokenspeed.runtime.execution.drafter.eagle_harness.config import (
    EagleHarnessConfig,
    chain_table_width,
    check_chain_cost_domain,
    first_rank_value,
)
from tokenspeed.runtime.execution.drafter.eagle_harness.cost import load_cost_data
from tokenspeed.runtime.execution.drafter.eagle_harness.device_chain import (
    DeviceChain,
)
from tokenspeed.runtime.execution.drafter.eagle_harness.device_policy import (
    DevicePolicy,
)
from tokenspeed.runtime.execution.drafter.eagle_harness.lanes import (
    Eagle4TreeLanes,
)
from tokenspeed.runtime.execution.drafter.eagle_harness.measurement import (
    CostMeasurement,
    MeasurePolicy,
)
from tokenspeed.runtime.execution.drafter.eagle_harness.round_stats import (
    RoundStats,
)
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.execution.tree_spec import TreeSpec, TreeSpecConfig
from tokenspeed.runtime.layers.attention.configs.base import resolve_cache_layer_types
from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
from tokenspeed.runtime.sampling.backends.base import (
    SamplingBackend,
    SpeculativeSamplingPools,
)
from tokenspeed.runtime.sampling.sampling_params import SamplingParams
from tokenspeed.runtime.sampling.tree_verify import DraftTreeQ
from tokenspeed.runtime.utils.common import maybe_inference_mode
from tokenspeed.runtime.utils.env import envs
from tokenspeed.runtime.utils.host_sync import allow_host_sync

if TYPE_CHECKING:
    from tokenspeed.runtime.models.llama_eagle4 import Eagle4LogitsOutput

__all__ = ["Eagle4", "EagleHarness"]

# Depth-0 weight of the root, above every child's path probability.
_ROOT_SCORE = 1.001


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _PendingRound:
    """A round between ``finish_round`` and ``complete_round``."""

    ctx: ForwardContext
    stream: torch.cuda.Stream  # the forward's stream, which joins the round
    bs: int  # live requests
    graph_bs: int | None  # the replayed decode graph's padded batch (None: eager)
    sampled: bool  # the round draws its children (planned depth)
    chained: bool  # the GPU launches its depths and finalize (DeviceChain)
    depth: int = 1  # last launched depth (host-launched rounds)


class Eagle4(Eagle):
    """The EAGLE drafter over EAGLE-4's candidate head, whose logits columns
    are each row's candidate tokens, drafting trees on ``Eagle4TreeLanes``."""

    @override
    def _make_tree_lanes(self, config: TreeSpecConfig, max_bs: int) -> Eagle4TreeLanes:
        draft_config = self.draft_model_runner.model.config
        # The draft layer's cache label, as its attention reads its window from it.
        layer_types = resolve_cache_layer_types(
            draft_config,
            num_layers=draft_config.num_hidden_layers,
            is_draft=True,
            draft_block_decode=False,
        )
        return Eagle4TreeLanes(
            config.topk,
            config.num_steps,
            max_bs,
            self.device,
            slides="sliding_attention" in layer_types,
        )

    @override
    def _score_candidates(
        self, logits_output: LogitsProcessorOutput
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores, columns = logprob_topk(
            logits_output.next_token_logits, self.draft_tree.topk
        )
        tokens = logits_output.candidate_ids.gather(-1, columns.long())
        return scores, tokens.to(columns.dtype)


class EagleHarness(Eagle4):
    """EAGLE drafter whose trees the EAGLE-Harness policy sizes per round.

    ``draft`` (inside the decode graph) runs the draft's first step and seeds
    the pool with the root's children and depth 1's lanes. The depths the
    policy decides on the device and the finalize writing the verify tree as
    the next round's inputs then run on the round stream, beside the
    forward's outputs and the next forward's staging, until ``complete_round``
    (in the next forward) joins them. Every buffer they share is persistent,
    and every count the kernels read is a device scalar, so one captured graph
    per (padded batch, depth) serves every round; a padded batch captures the
    depths its rounds can reach.

    With the device policy, a round on captured graphs is device-chained
    (``DeviceChain``): the host launches depth 1's graph behind the decode
    graph (``replay_decode_graph``), and each depth's graph launches the next
    one or the finalize on the GPU. Otherwise (eager rounds, exec mode 2)
    ``finish_round`` launches depth 1, the host each next depth once its
    decision lands, and the finalize in ``complete_round``.
    """

    def __init__(self, *args, harness: EagleHarnessConfig, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.harness = harness
        self.policy: DevicePolicy | MeasurePolicy | None = None
        # Exec mode 2: the cost measurement this drafter's rounds sample.
        self.measurement: CostMeasurement | None = None
        self._sampling_backend: SamplingBackend | None = None
        # Node pool and round buffers, allocated by bind_tree.
        self._pool: dict[str, torch.Tensor] = {}
        self._round: dict[str, torch.Tensor] = {}
        self._root_children: torch.Tensor | None = None
        # What _round["live_rows"] and ["sampled"] hold (written on change).
        self._live_rows = 0
        self._live_sampled = False
        # Recorded behind each depth the host launches (index 0: before depth 1).
        self._depth_done: list[torch.cuda.Event] = []
        # Runs a depth's decision beside its forward, which reads none of its outputs.
        self._decide_stream = torch.cuda.Stream(device=self.device)
        # Drafts a round beside the next forward's staging.
        self._round_stream = torch.cuda.Stream(device=self.device)
        # The device-launchable round graphs (the device policy's; else None).
        self._chain: DeviceChain | None = None
        # Whether the coming round draws its children (prepare_round_sampling).
        self._round_sampled = False
        # Recorded on the round stream behind the decode graph a round follows.
        self._target_done = torch.cuda.Event()
        # The padded batch whose round replay_decode_graph launched, until finish_round.
        self._launched_rows: int | None = None
        # The round finish_round started and complete_round finishes.
        self._pending: _PendingRound | None = None
        # (padded batch, depth) -> captured depth graph.
        self._depth_graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        # (padded batch, sampled) -> captured ranking and verify tree write.
        self._finalize_graphs: dict[tuple[int, bool], torch.cuda.CUDAGraph] = {}
        # Sampled trees: the verifier's pools, the round's sampling scalars, counter and q.
        self._pools: SpeculativeSamplingPools | None = None
        self._sampled: dict[str, torch.Tensor] = {}
        self._sampled_q: DraftTreeQ | None = None
        # Rank 0's round counts (TOKENSPEED_EAGLE_HARNESS_ROUND_STATS; None: off).
        interval = envs.TOKENSPEED_EAGLE_HARNESS_ROUND_STATS.get()
        self._round_stats: RoundStats | None = (
            RoundStats(interval)
            if interval > 0 and self.draft_model_runner.mapping.rank == 0
            else None
        )

    @override
    def bind_tree(self, tree_spec: TreeSpec) -> None:
        if self.draft_reads_token_history:
            raise NotImplementedError(
                "tree drafting does not support draft token history yet"
            )
        self.tree_spec = tree_spec
        config = tree_spec.config
        topk, depth = config.topk, config.num_steps
        max_bs = self.input_buffers.max_bs
        self._bind_tree_lanes(config, max_bs)
        logits_processor = self.draft_model_runner.model.logits_processor
        self.tree_vocab_size = logits_processor.config.vocab_size
        logits_processor.draft_top_k = topk
        # Children come from draft_topk; the head's argmax would go unread.
        logits_processor.do_argmax = False
        self.policy = self._make_policy(config.num_nodes, topk, depth)
        if isinstance(self.policy, DevicePolicy):
            self._chain = DeviceChain(max_bs, depth - 1, self.device)
        pool_nodes = max(1024, topk * topk * depth)
        self._allocate(max_bs, topk, depth, pool_nodes, max(64, topk * (depth + 1) + 1))
        if self.harness.sampled_trees:
            self._allocate_sampled(max_bs, topk, depth, config.num_nodes, pool_nodes)

    def _allocate_sampled(
        self, max_bs: int, topk: int, depth: int, num_nodes: int, pool_nodes: int
    ) -> None:
        """Sampled rounds' buffers (trees and chains) and the next round's q
        tables: one q slot for the root and one per lane row of every later
        depth."""
        device = self.device
        num_candidates = self.draft_model_runner.model.num_candidates
        num_slots = 1 + (depth - 1) * topk
        self._sampled_q = DraftTreeQ.zeros(
            max_bs, num_slots, num_candidates, pool_nodes, device
        )
        self._sampled = {
            "round": torch.zeros((1,), dtype=torch.int64, device=device),
            "temperature": torch.zeros((max_bs,), dtype=torch.float32, device=device),
            "top_k": torch.ones((max_bs,), dtype=torch.int32, device=device),
            "top_p": torch.ones((max_bs,), dtype=torch.float32, device=device),
        }
        self.runtime_states.init_draft_tree_q(num_nodes, num_slots, num_candidates)

    @override
    def bind_sampling_backend(self, sampling_backend: SamplingBackend) -> None:
        self._sampling_backend = sampling_backend
        if self.harness.sampled_trees:
            self._pools = sampling_backend.speculative_sampling_pools()

    @override
    def prepare_target_forward(self, ctx: ForwardContext) -> None:
        super().prepare_target_forward(ctx)
        if self.measurement is not None and ctx.num_extends == 0:
            # Inside the decode graph: the verify the measurement times starts here.
            self.measurement.verify_start.record()

    @override
    def rewrite_verify_outputs(
        self,
        ctx: ForwardContext,
        output_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
    ) -> None:
        """Exec mode 2: every request commits the root alone, as the fixed
        token (the context grows by one token a round), or as an EOS once the
        batch has its rounds."""
        measurement = self.measurement
        if measurement is None or ctx.num_extends > 0:
            return
        measurement.verify_end.record()
        n = self.spec_num_tokens
        rows = accept_lengths.shape[0]
        accept_lengths.fill_(1)
        output_tokens.view(rows, n)[:, 0].copy_(measurement.commit_token.expand(rows))
        path = self._sampling_backend.accepted_path(rows, n)
        path[:, 0] = 0
        path[:, 1:] = -1

    def _sample_children(
        self,
        logits_output: Eagle4LogitsOutput,
        tokens: torch.Tensor,
        scores: torch.Tensor,
        *,
        width: torch.Tensor | None,
        slot_base: int,
        cumu_start: torch.Tensor,
        launch: int,
    ) -> None:
        """Redraw the sampling requests' children of their expanded rows
        (``tokens`` / ``scores``: ``[bs, rows, K]``, each request's first
        ``rows`` forwarded rows, of which it expands the first ``width``; None:
        all) from their draft q, in place; greedy requests keep their top-K."""
        sampled = self._sampled
        bs, rows = tokens.shape[:2]
        logits, candidates = (
            output.view(bs, -1, output.shape[-1])[:, :rows]
            for output in (logits_output.next_token_logits, logits_output.candidate_ids)
        )
        eagle_harness_sample_children(
            logits,
            candidates,
            tokens,
            scores,
            sampled["temperature"],
            sampled["top_k"],
            sampled["top_p"],
            self._sampled_q.q,
            self._sampled_q.q_ids,
            self._sampled_q.slot,
            self._sampled_q.pos,
            width,
            slot_base,
            cumu_start,
            self.tree_vocab_size,
            self.harness.seed,
            sampled["round"],
            launch,
        )

    def round_depth(self, bs: int) -> int:
        """The deepest depth a round of ``bs`` live requests can launch: the
        width table's ``width - 2`` (depth 1 always runs), else the last."""
        width = chain_table_width(self.harness.chain_width_table, bs)
        return self.policy.tree_depth - 1 if width is None else max(width - 2, 1)

    def _make_policy(
        self, verify_num: int, topk: int, depth: int
    ) -> DevicePolicy | MeasurePolicy:
        """The round decisions: measured widths (exec mode 2, which then
        also starts its measurement), or the cost-data policy of the
        compute intensity, taken on the device."""
        harness = self.harness
        mapping = self.draft_model_runner.mapping
        # The harness files decide each round's collectives: the TP ranks take the first rank's read.
        file_group = None
        if mapping.attn.tp_size > 1:
            pg_manager.init_process_group(
                mapping.attn.tp_group, backend="gloo", name="eagle_harness_files"
            )
            file_group = pg_manager.get_process_group(
                "gloo", mapping.attn.tp_group, name="eagle_harness_files"
            )
        if harness.exec_mode == 2:
            if harness.eos_token_id is None:
                raise ValueError(
                    "EAGLE-Harness exec mode 2 ends a measured batch with an EOS "
                    "token; the model declares none"
                )
            self.measurement = CostMeasurement(
                run_dir=harness.data_path,
                tree_depth=depth,
                tp_group=(
                    pg_manager.get_device_process_group(mapping.attn.tp_group)
                    if mapping.attn.tp_size > 1
                    else None
                ),
                file_group=file_group,
                rank=mapping.rank,
                eos_token=harness.eos_token_id,
                device=self.device,
            )
            return MeasurePolicy(
                verify_num=verify_num,
                node_top_k=topk,
                tree_depth=depth,
                compute_intensity=harness.compute_intensity,
                device=self.device,
            )
        cost_dir = f"{harness.data_path}/cost"
        cost_data = first_rank_value(lambda: load_cost_data(cost_dir), file_group)
        if harness.compute_intensity == "chain_intensity":
            check_chain_cost_domain(
                sorted(cost_data),
                harness.chain_width_table,
                harness.max_requests,
                cost_dir,
            )
        return DevicePolicy(
            cost_data,
            verify_num=verify_num,
            node_top_k=topk,
            tree_depth=depth,
            compute_intensity=harness.compute_intensity,
            is_moe_target=harness.is_moe_target,
            exec_mode=harness.exec_mode,
            benchmark_config=f"{harness.data_path}/config/benchmark_config.json",
            chain_width_table=harness.chain_width_table,
            device=self.device,
            file_group=file_group,
        )

    def _allocate(
        self, max_bs: int, topk: int, depth: int, nodes: int, verify: int
    ) -> None:
        """The node pool and lists (``nodes`` nodes, work lists for ``verify``),
        initialized as EAGLE-4 does, and the round buffers ``draft`` and
        ``finish_round`` share."""
        device = self.device
        scores = torch.ones((max_bs, nodes), dtype=torch.float32, device=device)
        scores[:, 0] = _ROOT_SCORE
        prev = torch.full((max_bs, nodes), -1, dtype=torch.int64, device=device)
        prev[:, 1 : topk + 1] = 0
        work = torch.zeros((max_bs, verify + topk), dtype=torch.float32, device=device)
        work[:, 0] = _ROOT_SCORE
        sort = torch.zeros((max_bs, verify), dtype=torch.float32, device=device)
        sort[:, 0] = _ROOT_SCORE
        self._pool = {
            "tokens": torch.zeros((max_bs, nodes), dtype=torch.int64, device=device),
            "scores": scores,
            "prev": prev,
            "work": work,
            "sort": sort,
        }

        hidden_size = self.draft_model_runner.model.config.hidden_size
        dtype = self.draft_model_runner.model_config.dtype
        int64 = dict(dtype=torch.int64, device=device)
        float32 = dict(dtype=torch.float32, device=device)
        self._round = {
            # The frontier: its rows' tokens, path scores and input hidden rows.
            "tokens": torch.zeros((max_bs, topk), **int64),
            "scores": torch.zeros((max_bs, topk), **float32),
            "hidden": torch.zeros(
                (max_bs, topk, hidden_size), dtype=dtype, device=device
            ),
            # The accepted frontier (lane slot 0) and the lanes' positions and slots.
            "frontier": torch.zeros((max_bs,), **int64),
            "positions": torch.zeros((max_bs * topk,), **int64),
            "slot_starts": torch.zeros((max_bs,), **int64),
            # Frontier pool indices, ping-ponged between consecutive depths.
            "frontier_index": torch.zeros((2, max_bs, topk), **int64),
            # The round's requests' rows of the next round's inputs.
            "rows": torch.zeros((max_bs,), **int64),
            # Live requests, which a graph's batch statistics read past its padding.
            "live_rows": torch.zeros((1,), dtype=torch.int32, device=device),
            "sampled": torch.zeros((1,), dtype=torch.bool, device=device),
            # The root's first child's pool index, where depth 0's children start.
            "first_child": torch.ones((1,), dtype=torch.int32, device=device),
            # By depth parity: a depth's decision reads the previous depth's deltas.
            "work_deltas": torch.zeros((2, topk + 1), **float32),
            # The ranked pool: each request's best nodes and their scores.
            "top_scores": torch.zeros(
                (max_bs, self.policy.verify_num_limit), **float32
            ),
            "top_index": torch.zeros((max_bs, self.policy.verify_num_limit), **int64),
        }
        # Depth 1's frontier: the root's children, pool nodes 1..K.
        self._root_children = (torch.arange(topk, device=device) + 1).repeat(max_bs, 1)
        # Depth 1's lane masks: lane r sees its own slot r (bit r of word 0).
        seed = torch.zeros((topk, self.tree_lanes.mask_words), **int64)
        seed[:, :1] = (1 << torch.arange(topk, device=device))[:, None]
        self._tree_seed_lane_mask = seed
        self._depth_done = [torch.cuda.Event() for _ in range(depth)]

    # --- Round ---

    @override
    def draft(self, draft_input: EagleDraftInput) -> torch.Tensor:
        """The round's first step, the root's children and depth 1's lanes; the
        next round's inputs are the root alone until ``finish_round`` writes
        the tree."""
        bs = draft_input.accept_lengths.shape[0]
        k = self.lanes_per_request
        pool, state = self._pool, self._round

        next_tokens = self._last_verified_window(bs, draft_input)
        self.tree_spec.draft_parent_buf[:bs].copy_(self.tree_spec.chain_parent)

        frontier = self._accepted_frontier(bs, draft_input)
        narrowing = AcceptedPrefixPublisher(self.attn_backend, frontier)
        logits_output, _ = self._run_first_step(bs, draft_input, narrowing)

        # Depth 1: the root's K best children (head softmax, best first).
        top_p, top_tokens = logits_output.draft_topk
        if self.harness.sampled_trees:
            self._load_round_sampling(bs)
            self._sample_children(
                logits_output,
                top_tokens[:, None],
                top_p[:, None],
                width=None,
                slot_base=0,
                cumu_start=state["first_child"],
                launch=0,
            )
        pool["tokens"][:bs, 0] = next_tokens[:, 0]
        eagle_harness_update_node_data(
            pool["scores"][:bs],
            pool["tokens"][:bs],
            pool["work"][:bs],
            pool["sort"][:bs],
            top_p,
            top_tokens,
        )
        state["tokens"][:bs].copy_(top_tokens)
        state["scores"][:bs].copy_(top_p)
        state["hidden"][:bs].copy_(
            logits_output.hidden_states[:, None, :].expand(-1, k, -1)
        )
        state["frontier"][:bs].copy_(frontier)
        # The next forward refills the input buffers before the round completes.
        state["rows"][:bs].copy_(
            self.input_buffers.state_write_req_pool_indices_buf[:bs]
        )
        self._start_lanes(bs)
        return next_tokens

    def _load_round_sampling(self, bs: int) -> None:
        """The round's per-request sampling scalars (greedy requests at
        temperature 0) and a fresh draw counter; tensor-only, in the graph."""
        pools, sampled = self._pools, self._sampled
        pool_indices = self.input_buffers.req_pool_indices_buf[:bs]
        top_k = sampled["top_k"][:bs]
        torch.index_select(pools.top_k, 0, pool_indices, out=top_k)
        torch.index_select(pools.top_p, 0, pool_indices, out=sampled["top_p"][:bs])
        temperature = sampled["temperature"][:bs]
        torch.index_select(pools.temperature, 0, pool_indices, out=temperature)
        temperature.masked_fill_(top_k == 1, 0.0)
        sampled["round"].add_(1)

    @override
    def prepare_round_sampling(
        self, sampling_params_list: list[SamplingParams]
    ) -> None:
        self._round_sampled = self.harness.sampled_trees and any(
            params.top_k != 1 for params in sampling_params_list
        )

    @override
    def replay_decode_graph(
        self, graph: torch.cuda.CUDAGraph, *, bs: int, graph_bs: int
    ) -> None:
        """With device-chained rounds, replay the decode graph on the round
        stream and launch its round's depth 1 behind it: the forward's stream
        waits for the decode graph alone, so the verified-state commit, the
        outputs and the next forward's staging run beside the round."""
        if self._chain is None or bs == 0:
            graph.replay()
            return
        if self._launched_rows is not None:
            raise RuntimeError(
                "EAGLE-Harness decode graph replayed before finish_round took "
                "the round the last replay launched"
            )
        self._open_round(bs, graph_bs, num_extends=0)
        stream, round_stream = torch.cuda.current_stream(), self._round_stream
        round_stream.wait_stream(stream)
        with torch.cuda.stream(round_stream):
            graph.replay()
            self._target_done.record()
            # Its graph launches the rest of the round.
            self._depth_graphs[(graph_bs, 1)].replay()
        stream.wait_event(self._target_done)
        self._launched_rows = graph_bs

    @maybe_inference_mode()
    def finish_round(self, ctx: ForwardContext, graph_bs: int | None) -> None:
        """Start drafting the depths the policy decides, on the round stream:
        begin the round and launch depth 1 (unless the decode graph's replay
        did); a device-chained round launches the rest on the GPU, otherwise
        every next depth whose decision has landed launches here and the rest
        in ``complete_round``.

        ``graph_bs`` is the padded batch the decode graph replayed, whose depth
        graphs then replay too (None: the forward ran eagerly, and so do the
        depths). Batch statistics (work deltas, the verify ranking) read only
        the live rows. The round draws its children when a request samples.
        """
        bs = ctx.bs
        launched, self._launched_rows = self._launched_rows, None
        if bs == 0 or self.input_buffers.all_extends_mid_chunk:
            return
        if launched is None:
            self._open_round(bs, graph_bs, num_extends=ctx.num_extends)
        elif launched != graph_bs:
            raise RuntimeError(
                f"EAGLE-Harness decode graph at {launched} rows launched a round "
                f"the forward finishes at {graph_bs}"
            )
        chained = self._chain is not None and graph_bs is not None
        self._pending = pending = _PendingRound(
            ctx=ctx,
            stream=torch.cuda.current_stream(),
            bs=bs,
            graph_bs=graph_bs,
            sampled=self._round_sampled,
            chained=chained,
        )
        if launched is not None:
            return
        self._round_stream.wait_stream(pending.stream)
        with torch.cuda.stream(self._round_stream):
            if self.measurement is not None:
                self._measure_round(pending)
            elif chained:
                # Its graph launches the rest of the round.
                self._depth_graphs[(graph_bs, 1)].replay()
            else:
                # Depth 1 begins the round on the device; the policy always drafts it.
                self._launch_depth(pending, 1)
                self._advance(pending, pause_before_wait=True)

    def _open_round(self, bs: int, graph_bs: int | None, *, num_extends: int) -> None:
        """Before a round's depth 1, on the forward's stream: refuse it when it
        could reach a depth its padded batch did not capture, count it (exec
        mode 2: book its measurement), and write its live requests and sampled
        flag (on change)."""
        if graph_bs is not None:
            self._check_round_graphs(bs, graph_bs)
        measurement = self.measurement
        if measurement is not None:
            if num_extends > 0:
                measurement.on_extend()
            else:
                measurement.record_verify(bs)
        else:
            self.policy.next_round(bs)
            if self._round_stats is not None:
                width = chain_table_width(self.harness.chain_width_table, bs)
                line = self._round_stats.record(bs, width, num_extends > 0)
                if line is not None:
                    logger.info(line)
        state = self._round
        if bs != self._live_rows:
            state["live_rows"].fill_(bs)
            self._live_rows = bs
        if self._round_sampled != self._live_sampled:
            state["sampled"].fill_(self._round_sampled)
            self._live_sampled = self._round_sampled

    def _check_round_graphs(self, bs: int, graph_bs: int) -> None:
        """Refuse a round of ``bs`` requests on padded batch ``graph_bs`` that
        could reach a depth the batch did not capture."""
        depth = self.round_depth(bs)
        if (graph_bs, depth) not in self._depth_graphs:
            raise RuntimeError(
                f"EAGLE-Harness round of {bs} requests can draft depth {depth}, "
                f"which its padded batch {graph_bs} did not capture (round_depth)"
            )

    def _measure_round(self, pending: _PendingRound) -> None:
        """Exec mode 2: draft the whole round at the measured widths between
        drained GPUs, sample its draft time, and write the batch's samples
        once it has all its rounds."""
        measurement, policy = self.measurement, self.policy
        bs = pending.bs
        with allow_host_sync("exec mode 2 measures between drained GPUs"):
            average_cache_len = int(self._round["frontier"][:bs].sum()) // bs
            if measurement.measured_batch_size is not None:
                policy.next_widths(bs, measurement.measured_batch_size)
            policy.write_round()
        measurement.begin_draft()
        # Measured widths never wait for the device: every depth drafts now.
        self._launch_depth(pending, 1)
        self._advance(pending, pause_before_wait=True)
        measurement.end_draft(
            bs, policy.draft_width, policy.verify_width_now, average_cache_len
        )
        if measurement.complete():
            test_dir = measurement.write_model_records()
            logger.info(f"EAGLE-Harness cost measurement: wrote {test_dir}")

    @override
    @maybe_inference_mode()
    def complete_round(self) -> None:
        """Draft the rest of the pending round, then rank its pool and write
        its verify tree into the next round's inputs (a device-chained round
        does all this on the GPU), and order the forward's stream after it.

        The rest runs on the round stream, so the inputs and attention
        metadata the next forward queued on the forward's stream since
        ``finish_round`` run beside the depths instead of between them.
        """
        pending = self._pending
        if pending is None:
            return
        self._pending = None
        if not pending.chained:
            with torch.cuda.stream(self._round_stream):
                self._advance(pending, pause_before_wait=False)
                if pending.graph_bs is not None:
                    self._finalize_graphs[(pending.graph_bs, pending.sampled)].replay()
                else:
                    self._write_verify_tree(pending.bs, pending.sampled)
        pending.stream.wait_stream(self._round_stream)

    def _advance(self, pending: _PendingRound, *, pause_before_wait: bool) -> None:
        """Launch each next depth the policy decides to draft, until the round
        stops (or, with ``pause_before_wait``, until a decision has not landed
        yet). A depth's decision runs at its start beside its forward, so the
        host learns whether to launch the next depth while the GPU drafts it;
        a width table round's depths are known up front and launch at once."""
        table_width = chain_table_width(self.harness.chain_width_table, pending.bs)
        if self.measurement is None and table_width is not None:
            # The table fixes the round on the host: no decision to wait for.
            while pending.depth < self.round_depth(pending.bs):
                pending.depth += 1
                self._launch_depth(pending, pending.depth)
            return
        policy = self.policy
        while True:
            depth = pending.depth
            go = policy.go_ready(depth + 1)
            if go is None:
                if pause_before_wait:
                    return
                go = self._wait_decision(depth)
            if not go:
                return
            pending.depth = depth + 1
            self._launch_depth(pending, depth + 1)

    def _wait_decision(self, depth: int) -> bool:
        """Whether depth ``depth + 1`` drafts: once ``depth`` has started, its
        decision lands within microseconds, so poll the go flag until then."""
        self._depth_done[depth - 1].synchronize()
        done = self._depth_done[depth]
        while True:
            # Queried first: a finished depth has published its decision.
            finished = done.query()
            go = self.policy.go_ready(depth + 1)
            if go is not None:
                return go
            if finished:
                raise RuntimeError(
                    f"EAGLE-Harness round has no decision after depth {depth}"
                )

    def _launch_depth(self, pending: _PendingRound, depth: int) -> None:
        """Draft ``depth`` and its decision (its captured graph, or eagerly),
        and mark where it ends."""
        if depth == 1:
            self._depth_done[0].record()
        if pending.graph_bs is not None:
            self._depth_graphs[(pending.graph_bs, depth)].replay()
        else:
            ctx = pending.ctx
            lanes = self._lane_context(
                pending.bs,
                ctx.global_num_tokens,
                ctx.global_bs,
                ctx.all_decode_or_idle,
            )
            with self.attn_backend.override_num_extends(0):
                self._draft_depth(pending.bs, lanes, depth, decide=True, chain=False)
        self._depth_done[depth].record()

    def _start_lanes(self, rows: int) -> None:
        """Depth 1's lane positions and slots (the accepted frontier) and lane
        masks; the lanes take this round's tables."""
        state, lanes = self._round, self.tree_lanes
        frontier = state["frontier"][:rows]
        lanes.set_frontier(rows, frontier)
        k, words = self.lanes_per_request, lanes.mask_words
        state["positions"][: rows * k].view(rows, k).copy_(
            frontier[:, None].expand(-1, k)
        )
        state["slot_starts"][:rows].copy_(frontier)
        lanes.lane_mask[: rows * k * words].view(rows, k, words).copy_(
            self._tree_seed_lane_mask
        )

    def _queue_work_deltas(self, rows: int, depth: int) -> None:
        """Depth ``depth``'s work deltas over the live rows among ``rows``, at
        the depths whose slot can ask for them (its work size; 0: none)."""
        policy = self.policy
        if not policy.deltas_depth(depth):
            return
        eagle_harness_work_deltas(
            self._pool["work"][:rows],
            self._round["work_deltas"][depth % 2],
            self._round["live_rows"],
            policy.work_sizes[depth : depth + 1],
            policy.past_verify_num,
        )

    @override
    @maybe_inference_mode()
    def capture_round_graphs(
        self, graph_bs: int, *, stream: torch.cuda.Stream, pool: tuple[int, int]
    ) -> None:
        """Capture each draft depth at the decode graph's padded batch: its
        widths, pool sizes and work list sizes are device slots, so one graph
        per depth serves every round. Only the depths reachable by the rounds
        that replay it: of floor + 1 to ``graph_bs`` live requests
        (``decode_graph_floors``; floor 0 for a padded batch it does not list).
        A device chain's depths launch the round's next graph after their
        decision; depths 2 and on and the finalize are instantiated for device
        launch."""
        floor = dict(self.harness.decode_graph_floors).get(graph_bs, 0)
        depths = max(self.round_depth(bs) for bs in range(floor + 1, graph_bs + 1))
        lanes = self._lane_context(graph_bs, None, None, True)
        chain = self._chain
        with torch.cuda.stream(stream), self.attn_backend.override_num_extends(0):
            # Warm up on a full round, without the decisions whose state the rounds own.
            self.policy.write_full_round()
            self._start_lanes(graph_bs)
            for depth in range(1, depths + 1):
                self._draft_depth(graph_bs, lanes, depth, decide=False, chain=False)
                torch.cuda.synchronize()
                # The host launches depth 1 (an ordinary graph); the GPU, the rest.
                graph = torch.cuda.CUDAGraph(keep_graph=chain is not None and depth > 1)
                with torch.cuda.graph(graph, pool=pool, stream=stream):
                    self._draft_depth(
                        graph_bs, lanes, depth, decide=True, chain=chain is not None
                    )
                self._depth_graphs[(graph_bs, depth)] = graph
            samples = (False, True) if self.harness.sampled_trees else (False,)
            for sampled in samples:
                graph = torch.cuda.CUDAGraph(keep_graph=chain is not None)
                with torch.cuda.graph(graph, pool=pool, stream=stream):
                    self._write_verify_tree(graph_bs, sampled)
                self._finalize_graphs[(graph_bs, sampled)] = graph
            if chain is not None:
                chain.instantiate(
                    graph_bs,
                    [self._depth_graphs[(graph_bs, d)] for d in range(2, depths + 1)],
                    [self._finalize_graphs[(graph_bs, sampled)] for sampled in samples],
                    stream,
                )
            if self.harness.sampled_trees:
                # Serving's draws start at round 1 however many warmups capture ran.
                self._sampled["round"].zero_()

    @override
    def release_round_graphs(self) -> None:
        if self._chain is not None:
            self._chain.release()
        self._depth_graphs.clear()
        self._finalize_graphs.clear()

    def _lane_context(
        self,
        bs: int,
        global_num_tokens: list[int] | None,
        global_bs: list[int] | None,
        all_decode_or_idle: bool,
    ) -> ForwardContext:
        """The forward context of a depth's ``K`` lanes per request."""
        k = self.lanes_per_request
        return ForwardContext(
            bs=bs,
            num_extends=0,
            output_layout=ForwardOutputLayout(0, 0, bs, k),
            attn_backend=self.attn_backend,
            token_to_kv_pool=self.token_to_kv_pool,
            input_num_tokens=bs * k,
            forward_mode=ForwardMode.DECODE,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            global_num_tokens=global_num_tokens,
            global_bs=global_bs,
            all_decode_or_idle=all_decode_or_idle,
        )

    def _draft_depth(
        self, bs: int, lanes: ForwardContext, depth: int, *, decide: bool, chain: bool
    ) -> None:
        """Forward depth ``depth``'s frontier (lane slots ``(depth - 1) * K``
        on) and append the ``K`` best children of each row's ``widths[depth]``
        best frontier nodes to the pool from ``cumu_starts[depth]`` on; leaves
        the next frontier in the round buffers. With ``decide``, the decision
        after ``depth`` (which reads only the previous depth's work deltas)
        runs beside the forward, after the round's start at depth 1; with
        ``chain`` (a captured depth of a device chain), the launch of the
        round's next graph follows that decision there, off the forward's
        path, and starts once this depth's graph has ended.

        All ``K`` lanes forward; past the width their outputs are dropped, and
        their K/V (in their own lane slots) is no node's ancestor. The lane
        masks this depth's forward reads come from the previous depth's
        update (depth 1's from ``_start_lanes``).
        """
        k, policy = self.lanes_per_request, self.policy
        pool, state, tree_lanes = self._pool, self._round, self.tree_lanes
        width = policy.widths[depth : depth + 1]
        cumu_start = policy.cumu_starts[depth : depth + 1]
        stream, decide_stream = torch.cuda.current_stream(), self._decide_stream
        if decide:
            decide_stream.wait_stream(stream)
            with torch.cuda.stream(decide_stream):
                if depth == 1:
                    policy.begin_round(
                        state["live_rows"], state["frontier"][:bs], state["sampled"]
                    )
                    # Depth 0 weighs the root (one node) against its K children.
                    self._queue_work_deltas(bs, 0)
                policy.decide(depth, state["work_deltas"][(depth - 1) % 2])
                if chain:
                    self._chain.end_depth(bs, depth, policy.go, state["sampled"])
        tree_lanes.active = True
        self.attn_backend.publish_draft_step_locations(
            cache_start=state["slot_starts"][:bs], num_tokens=k
        )
        logits_output = self.draft_model_runner.forward(
            ctx=lanes,
            input_ids=state["tokens"][:bs].view(-1),
            positions=state["positions"][: bs * k],
            captured_hidden_states=state["hidden"][:bs].view(bs * k, -1),
            spec_step_idx=depth,
        )
        tree_lanes.active = False
        if decide and depth == 1:
            # Depth 1's counts come from the round's start.
            stream.wait_stream(decide_stream)
        # The lanes' children; each request expands its first ``width`` rows.
        child_p, child_tokens = (
            output.view(bs, k, k) for output in logits_output.draft_topk
        )
        if self.harness.sampled_trees:
            self._sample_children(
                logits_output,
                child_tokens,
                child_p,
                width=width,
                slot_base=1 + (depth - 1) * k,
                cumu_start=cumu_start,
                launch=depth,
            )
        # Children sit one depth below the frontier the forward expanded.
        expanded = (
            self._root_children[:bs]
            if depth == 1
            else state["frontier_index"][(depth + 1) % 2][:bs]
        )
        words = tree_lanes.mask_words
        eagle_harness_draft_depth_update(
            child_p,
            child_tokens,
            expanded,
            logits_output.hidden_states.view(bs, k, -1),
            state["tokens"][:bs],
            state["scores"][:bs],
            state["hidden"][:bs],
            state["frontier_index"][depth % 2][:bs],
            pool["tokens"][:bs],
            pool["scores"][:bs],
            pool["prev"][:bs],
            pool["work"][:bs],
            pool["sort"][:bs],
            state["positions"][: bs * k].view(bs, k),
            state["slot_starts"][:bs],
            tree_lanes.lane_mask[: bs * k * words].view(bs, k, words),
            width,
            depth,
            cumu_start,
            policy.past_verify_num,
            tree_lanes.num_slots,
        )
        self._queue_work_deltas(bs, depth)
        if decide and depth > 1:
            stream.wait_stream(decide_stream)

    def _write_verify_tree(self, bs: int, sampled: bool) -> None:
        """Rank the pool of the live rows among ``bs``, size the verify, and
        write the verify tree as the next round's inputs: the best nodes in
        ancestor-first score order, root first; unused slots repeat the root
        with parent -1 and are never accepted.

        All on the device: the verify size reads the previous round's ranked
        scores (the first round its own), and this round's wait there for the
        next.
        """
        pool, policy, state = self._pool, self.policy, self._round
        live_rows = state["live_rows"]
        top_scores, top_index = state["top_scores"][:bs], state["top_index"][:bs]
        eagle_harness_stable_topk_desc(
            pool["scores"][:bs],
            live_rows,
            policy.cumu_count,
            policy.verify_count,
            top_scores,
            top_index,
        )
        policy.select_verify(top_scores)
        eagle_harness_write_verify_tree(
            top_index,
            pool["tokens"][:bs],
            pool["prev"][:bs],
            state["rows"][:bs],
            self.runtime_states.future_input_map,
            self.runtime_states.future_parent_map,
            live_rows,
            policy.verify_num,
        )
        if sampled:
            # Only sampling rows read the tables; deterministic nodes carry slot -1.
            draft_q, round_q = self.runtime_states.future_draft_q, self._sampled_q
            eagle_harness_publish_draft_q(
                top_index,
                round_q.slot[:bs],
                round_q.pos[:bs],
                round_q.q[:bs],
                round_q.q_ids[:bs],
                state["rows"][:bs],
                draft_q.q,
                draft_q.q_ids,
                draft_q.slot,
                draft_q.pos,
                live_rows,
                policy.verify_num,
            )
