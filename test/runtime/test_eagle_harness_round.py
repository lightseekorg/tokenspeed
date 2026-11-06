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

"""EAGLE-Harness rounds across forwards on a GPU: a real drafter, device
policy (on synthetic cost data) and round kernels, with a weightless draft
model and an attention backend without a KV cache standing in for the rest.

``draft`` seeds a round, which drafts on the round stream beside the next
forward's staging until ``complete_round`` joins it: an eager round has
``finish_round`` and the host launch each depth once the device's decision
lands, a round on the graphs ``capture_round_graphs`` records launches them on
the GPU, behind the decode graph (``replay_decode_graph``) or from
``finish_round``."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest  # noqa: E402
import tokenspeed_scheduler as ts  # noqa: E402
import torch  # noqa: E402
from ci_system.ci_register import register_cuda_ci  # noqa: E402
from transformers import LlamaConfig  # noqa: E402

register_cuda_ci(
    est_time=10,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="the eagle_harness policy kernels are NVIDIA-only",
)

from test.runtime.scheduler_driver import advance_scheduler  # noqa: E402

from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.engine.scheduler_utils import make_config  # noqa: E402
from tokenspeed.runtime.execution.context import ForwardContext  # noqa: E402
from tokenspeed.runtime.execution.drafter.eagle import EagleDraftInput  # noqa: E402
from tokenspeed.runtime.execution.drafter.eagle_harness.config import (  # noqa: E402
    ChainWidthEntry,
    ChainWidthTable,
    EagleHarnessConfig,
    chain_table_width,
)
from tokenspeed.runtime.execution.drafter.eagle_harness.cost import (  # noqa: E402
    load_cost_data,
)
from tokenspeed.runtime.execution.drafter.eagle_harness.device_policy import (  # noqa: E402
    DevicePolicy,
)
from tokenspeed.runtime.execution.drafter.eagle_harness.drafter import (  # noqa: E402
    EagleHarness,
)
from tokenspeed.runtime.execution.drafter.eagle_harness.round_stats import (  # noqa: E402
    RoundStats,
)
from tokenspeed.runtime.execution.forward_batch_info import (  # noqa: E402
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.execution.input_buffer import InputBuffers  # noqa: E402
from tokenspeed.runtime.execution.output_layout import (  # noqa: E402
    ForwardOutputLayout,
)
from tokenspeed.runtime.execution.runtime_states import RuntimeStates  # noqa: E402
from tokenspeed.runtime.execution.tree_spec import (  # noqa: E402
    TreeSpec,
    TreeSpecConfig,
)
from tokenspeed.runtime.layers.attention.backends.base import (  # noqa: E402
    AttentionBackend,
)
from tokenspeed.runtime.models.llama_eagle4 import (  # noqa: E402
    Eagle4CandidateLogitsProcessor,
    Eagle4LogitsOutput,
)
from tokenspeed.runtime.sampling.sampling_params import SamplingParams  # noqa: E402
from tokenspeed.runtime.utils.env import envs  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# Trees of K children per node, STEPS steps, NODES verified; POOL slots, MAX_BS a round.
K, STEPS, NODES, POOL, MAX_BS = 8, 6, 32, 16, 8
# Chains: STEPS and NODES.
CHAIN_STEPS, CHAIN_NODES = 15, 16
HIDDEN, VOCAB = 64, 4096
# GPU sleeps: the next forward's staging, and a launch or draft forward that runs late.
_STAGING_CYCLES, _LAG_CYCLES = 200_000_000, 2_000_000


@dataclass(frozen=True)
class _Shape:
    """A drafter's rounds: ``k`` children per node, ``steps`` steps, ``nodes``
    verified, sized by the policy of ``intensity`` and a chain's width table;
    the decode graphs' (padded batch, floor) buckets."""

    k: int
    steps: int
    nodes: int
    intensity: str
    table: ChainWidthTable
    graph_floors: tuple[tuple[int, int], ...]


_TREE = _Shape(K, STEPS, NODES, "high_intensity", (), ())


def _chain(table: ChainWidthTable, graph_floors: tuple[tuple[int, int], ...]) -> _Shape:
    return _Shape(1, CHAIN_STEPS, CHAIN_NODES, "chain_intensity", table, graph_floors)


def _write_cost_data(root: str, *, chain: bool = False) -> None:
    """Cost data at batch sizes 1, 2, 4 and 8, three 512-token context
    intervals each; from batch 4 the verify cost jumps at width 6, so those
    batches draft narrower depths (a chain's: verify widths 1, 2, 4, ..., 16
    and one draft width)."""
    llm = [0, 1, 3, 5, 7, 9, 11, 13, 15] if chain else [0, 7, 15, 31]
    slm = [0] if chain else [0, 1, 3, 7]
    for bs in (1, 2, 4, 8):
        for model, widths, base, slope in (
            ("model_llm", llm, 4000.0, 0.012),
            ("model_slm", slm, 170.0, 0.01),
        ):
            path = os.path.join(root, "cost", f"bs_{bs}", model)
            os.makedirs(path)
            for interval in range(3):
                v0 = base * (1 + 0.15 * (bs - 1)) * (1 + 0.05 * interval)
                values = [
                    v0 * (1 + slope * w * (2 * bs - 1) + 0.3 * (w >= 5 and bs >= 4))
                    for w in widths
                ]
                with open(
                    os.path.join(path, f"model_interval_{interval}.json"), "w"
                ) as f:
                    json.dump(
                        {
                            "value": values,
                            "index": widths,
                            "ratio": [v / values[0] for v in values],
                            "cache_length_step": 512,
                        },
                        f,
                    )


class _TimedPolicy(DevicePolicy):
    """The device policy, its launches landing late (a GPU sleep on the
    launching stream first) or in program order (the device drained before
    and after each, whatever the drafter's streams wait for)."""

    def __init__(self, *args, late: bool, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.late = late

    def _timed(self, launch: Callable[..., None], *args: torch.Tensor | int) -> None:
        if self.late:
            torch.cuda._sleep(_LAG_CYCLES)
        else:
            torch.cuda.synchronize()
        launch(*args)
        if not self.late:
            torch.cuda.synchronize()

    def begin_round(
        self, live_rows: torch.Tensor, frontier: torch.Tensor, sampled: torch.Tensor
    ) -> None:
        self._timed(super().begin_round, live_rows, frontier, sampled)

    def decide(self, depth: int, work_deltas: torch.Tensor) -> None:
        self._timed(super().decide, depth, work_deltas)

    def select_verify(self, top_scores: torch.Tensor) -> None:
        self._timed(super().select_verify, top_scores)


class _DraftModel:
    """The EAGLE-4 draft model's parts the drafter configures: its config and
    candidate logits processor (the weights stay out)."""

    def __init__(self, mapping: Mapping) -> None:
        self.config = LlamaConfig(vocab_size=VOCAB, hidden_size=HIDDEN)
        self.logits_processor = Eagle4CandidateLogitsProcessor(
            self.config,
            tp_rank=mapping.lm_head.tp_rank,
            tp_size=mapping.lm_head.tp_size,
            tp_group=mapping.lm_head.tp_group,
        )

    def get_hot_token_id(self) -> None:
        return None


@dataclass(frozen=True)
class _DraftModelConfig:
    dtype: torch.dtype
    requires_request_token_history: bool


class _DraftRunner:
    """Stands in for the draft ModelRunner (no weights): each output row's K
    best children and their probabilities are a fixed function of its input
    token, sharper for some tokens than for others, so rounds stop at
    different depths. A forward takes ``forward_cycles`` of GPU time."""

    def __init__(self, forward_cycles: int) -> None:
        self.forward_cycles = forward_cycles
        # Forwards run (eagerly: a replayed graph runs none).
        self.forwards = 0
        self.device = torch.device("cuda")
        self.mapping = Mapping(rank=0, world_size=1)
        self.model = _DraftModel(self.mapping)
        self.model_config = _DraftModelConfig(
            dtype=torch.bfloat16, requires_request_token_history=False
        )

    def forward(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        captured_hidden_states: torch.Tensor,
        spec_step_idx: int,
    ) -> Eagle4LogitsOutput:
        self.forwards += 1
        torch.cuda._sleep(self.forward_cycles)
        rows, hidden = input_ids.long(), captured_hidden_states
        if ctx.gather_ids is not None:
            rows, hidden = rows[ctx.gather_ids], hidden[ctx.gather_ids]
        k = self.model.logits_processor.draft_top_k
        children = torch.arange(1, 2 * k + 1, device=rows.device)
        sharpness = 0.3 + 0.4 * (rows % 7)[:, None]
        # Contiguous rows, as the candidate head allocates its top-k.
        probs = torch.softmax(-sharpness * children, dim=-1)[:, :k].contiguous()
        tokens = (rows[:, None] * 31 + children[:k] * 97) % VOCAB
        return Eagle4LogitsOutput(
            next_token_logits=probs,
            hidden_states=hidden.clone(),
            draft_topk=(probs, tokens),
        )


class _NoCacheBackend(AttentionBackend):
    """The draft attention backend without a KV cache: it publishes no write
    locations (the weightless draft model attends nothing)."""

    def __init__(self) -> None:
        self._init_pool_binding()

    def publish_draft_step_locations(
        self, cache_start: torch.Tensor, num_tokens: int
    ) -> None:
        pass


@dataclass
class _Rounds:
    """A drafter and the buffers the executor fills for its rounds."""

    drafter: EagleHarness
    buffers: InputBuffers
    states: RuntimeStates
    nodes: int  # verified per request
    # The padded batches whose round graphs replay, ascending (empty: rounds run eagerly).
    ladder: tuple[int, ...] = ()

    def capture(self, ladder: tuple[int, ...]) -> None:
        """(Re)capture the round graphs at each padded batch of ``ladder``,
        largest first, as the executor does after each decode graph; later
        rounds replay those of the smallest padded batch they fit."""
        drafter = self.drafter
        drafter.release_round_graphs()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        pool = torch.cuda.graph_pool_handle()
        for graph_bs in sorted(ladder, reverse=True):
            drafter.capture_round_graphs(graph_bs, stream=stream, pool=pool)
        torch.cuda.current_stream().wait_stream(stream)
        self.ladder = tuple(sorted(ladder))

    def draft(
        self, bs: int, rng: random.Random, slots: list[int] | None = None
    ) -> list[int]:
        """Fill a decode round's inputs (``slots``, else random ones, and
        random contexts, verified tokens and accept lengths) and draft its
        first step; returns the slots."""
        if slots is None:
            slots = rng.sample(range(POOL), bs)
        device = self.drafter.device
        slot_rows = torch.tensor(slots, dtype=torch.int64, device=device)
        self.buffers.req_pool_indices_buf[:bs] = slot_rows
        self.buffers.state_write_req_pool_indices_buf[:bs] = slot_rows
        self.states.valid_cache_lengths[slot_rows] = torch.tensor(
            [rng.randint(64, 1500) for _ in slots], dtype=torch.int32, device=device
        )
        nodes = self.nodes
        tokens = [rng.randrange(VOCAB) for _ in range(bs * nodes)]
        generator = torch.Generator().manual_seed(rng.randrange(2**31))
        self.drafter.draft(
            EagleDraftInput(
                input_num_tokens=bs * nodes,
                num_extends=0,
                forward_mode=ForwardMode.DECODE,
                base_model_output=torch.tensor(
                    tokens, dtype=torch.int32, device=device
                ),
                accept_lengths=torch.tensor(
                    [rng.randint(1, nodes) for _ in slots],
                    dtype=torch.int32,
                    device=device,
                ),
                base_out_hidden_states=torch.randn(
                    (bs * nodes, HIDDEN), generator=generator
                ).to(device, torch.bfloat16),
                all_decode_or_idle=True,
            )
        )
        return slots

    def finish_round(self, bs: int) -> None:
        drafter = self.drafter
        drafter.prepare_round_sampling([SamplingParams(temperature=0.0)] * bs)
        drafter.finish_round(
            ForwardContext(
                attn_backend=drafter.attn_backend,
                token_to_kv_pool=None,
                bs=bs,
                num_extends=0,
                input_num_tokens=bs * self.nodes,
                forward_mode=ForwardMode.DECODE,
                output_layout=ForwardOutputLayout(0, 0, bs, self.nodes),
                capture_hidden_mode=CaptureHiddenMode.LAST,
                all_decode_or_idle=True,
            ),
            self.graph_bs(bs),
        )

    def graph_bs(self, bs: int) -> int | None:
        """The padded batch whose round graphs a round of ``bs`` replays."""
        return next((graph_bs for graph_bs in self.ladder if graph_bs >= bs), None)

    def verify_tree(self, slots: list[int]) -> torch.Tensor:
        """The next round's verify trees of ``slots`` (tokens, then parents),
        read on the current stream."""
        index = torch.tensor(slots, device=self.drafter.device)
        return torch.cat(
            [
                self.states.future_input_map[index],
                self.states.future_parent_map[index],
            ],
            dim=1,
        )


def _rounds(
    root: str, *, late: bool | None, forward_cycles: int, shape: _Shape
) -> _Rounds:
    """A drafter of ``shape`` as the executor builds and binds it, on cost
    data at ``root``; ``late`` times its policy's launches (None: as served)."""
    nodes = shape.nodes
    states = RuntimeStates(
        req_pool_size=POOL, vocab_size=VOCAB, output_length=nodes, device="cuda"
    )
    states.init_draft_trees(nodes)
    buffers = InputBuffers(
        max_bs=MAX_BS,
        max_num_tokens=MAX_BS * nodes,
        state_write_padding_pool_index=POOL,
    )
    harness = EagleHarnessConfig(
        exec_mode=0,
        data_path=root,
        compute_intensity=shape.intensity,
        is_moe_target=False,
        sampled_trees=False,
        seed=0,
        eos_token_id=None,
        chain_width_table=shape.table,
        max_requests=MAX_BS,
        decode_graph_floors=shape.graph_floors,
    )
    drafter = EagleHarness(
        spec_num_tokens=nodes,
        spec_num_steps=shape.steps,
        draft_model_runner=_DraftRunner(forward_cycles),
        runtime_states=states,
        input_buffers=buffers,
        attn_backend=_NoCacheBackend(),
        token_to_kv_pool=None,
        vocab_size=VOCAB,
        harness=harness,
    )
    drafter.bind_tree(
        TreeSpec(
            TreeSpecConfig(topk=shape.k, num_steps=shape.steps, num_nodes=nodes),
            max_bs=MAX_BS,
            device=drafter.device,
        )
    )
    if late is not None:
        drafter.policy = _TimedPolicy(
            load_cost_data(os.path.join(root, "cost")),
            verify_num=nodes,
            node_top_k=shape.k,
            tree_depth=shape.steps,
            compute_intensity=harness.compute_intensity,
            is_moe_target=False,
            exec_mode=0,
            benchmark_config=os.path.join(root, "config", "benchmark_config.json"),
            file_group=None,
            chain_width_table=harness.chain_width_table,
            device=drafter.device,
            late=late,
        )
    return _Rounds(drafter, buffers, states, nodes)


def _decode_batch(bs: int, width: int):
    """The scheduler's decode batch of ``bs`` requests, ``width`` tokens
    each, once their prefill landed."""
    scheduler = ts.Scheduler(
        make_config(
            num_device_pages=65,
            max_scheduled_tokens=bs * width,
            max_batch_size=bs,
            prefix_granularity=4,
            num_host_pages=0,
            disable_l2_cache=True,
            enable_l3_storage=False,
            role="fused",
            num_snapshot_pages=1,
            host_group_packing=(),
            max_retracted_requests=0,
            host_write_policy="write_through",
            storage_config=None,
            decode_input_tokens=width,
            cache_groups=[
                ts.CacheGroupConfig(
                    group_id="history",
                    block_granularity=4,
                    total_pages=65,
                    attn_kind=ts.AttnKind.Full,
                )
            ],
        )
    )
    for i in range(bs):
        spec = ts.RequestSpec()
        spec.request_id = f"r{i}"
        spec.tokens = [1, 2, 3]
        spec.max_new_tokens = 8
        scheduler.submit_requests([spec])
    (prefill,) = [
        op for op in scheduler.next_execution_plan().forward if op.request_ids
    ]
    results = []
    for request_id in prefill.request_ids:
        result = ts.ForwardEvent.ExtendResult()
        result.request_id = request_id
        result.tokens = [5]
        results.append(result)
    advance_scheduler(scheduler, results)
    (decode,) = [op for op in scheduler.next_execution_plan().forward if op.request_ids]
    assert decode.num_extends() == 0 and max(decode.request_pool_indices) < POOL
    return decode


def _depths(parents: torch.Tensor) -> list[int]:
    """Each node's depth in a verify tree (its parents' row)."""
    depths = []
    for parent in parents.tolist():
        depths.append(0 if parent < 0 else depths[parent] + 1)
    return depths


@pytest.mark.parametrize("timing", ["late_decisions", "slow_drafts", "graphs"])
@torch.inference_mode()
def test_rounds_write_the_same_trees_however_their_streams_interleave(
    tmp_path, timing: str
) -> None:
    """The decisions run on their own stream beside the depths they size.
    Rounds whose decisions (and ranking) land late, whose draft forwards run
    long, or which replay their captured graphs (late decisions too, captured
    before the first round and again between rounds) write the verify trees
    of eager rounds run in program order, each read on the round's stream as
    soon as ``complete_round`` returns: a depth reads its slots only after the
    decision writing them, a decision reads the previous depth's work deltas
    only after that depth wrote them, the next forward waits for the tree, and
    a capture's warm-up changes no later round."""
    _write_cost_data(str(tmp_path))
    root = str(tmp_path)
    timed = _rounds(
        root,
        late=True if timing != "slow_drafts" else None,
        forward_cycles=_LAG_CYCLES if timing == "slow_drafts" else 0,
        shape=_TREE,
    )
    ordered = _rounds(root, late=False, forward_cycles=0, shape=_TREE)
    # Batch sizes across the cost tiers, so depth 1's width changes between rounds.
    batch_sizes = [1, 4, 2, 8, 1, 5, 3, 8, 2, 6, 1, 7]
    captures = {0, 6} if timing == "graphs" else set()
    trees = []
    for rounds in (timed, ordered):
        rng = random.Random(0)
        trees.append([])
        for i, bs in enumerate(batch_sizes):
            if rounds is timed and i in captures:
                rounds.capture((MAX_BS,))
            slots = rounds.draft(bs, rng)
            rounds.finish_round(bs)
            rounds.drafter.complete_round()
            trees[-1].append(rounds.verify_tree(slots))
    depths = set()
    for timed_tree, ordered_tree in zip(*trees):
        assert torch.equal(timed_tree, ordered_tree)
        depths.update(max(_depths(row[NODES:])) for row in ordered_tree)
    # The rounds stop at different depths, some at the last one.
    assert len(depths) > 2 and STEPS - 1 in depths


@pytest.mark.parametrize("graphs", [False, True])
@torch.inference_mode()
def test_a_round_drafts_beside_the_next_forwards_staging(tmp_path, graphs) -> None:
    """A round, on the round stream, does not wait for what the next forward
    queued on the forward's stream after ``finish_round`` (its input fill),
    and that forward's load reads the round's verify trees; eagerly or
    device-chained."""
    _write_cost_data(str(tmp_path))
    rounds = _rounds(str(tmp_path), late=None, forward_cycles=0, shape=_TREE)
    if graphs:
        rounds.capture((MAX_BS,))
    buffers, states = rounds.buffers, rounds.states
    rng = random.Random(1)
    # The next forward, its requests reordered: it refills the buffers the round drafted from.
    op = _decode_batch(4, NODES)
    order = list(op.request_pool_indices)
    # A first round loads the kernels (a lazy module load synchronizes).
    for staging in (False, True):
        states.future_input_map.fill_(-1)
        slots = rounds.draft(4, rng, order[::-1])
        rounds.finish_round(4)
        total = len(order) * NODES
        decode_input_ids = buffers.fill_input_buffers(op, states, total)
        if staging:
            torch.cuda._sleep(_STAGING_CYCLES)
        rounds.drafter.complete_round()
    round_stream = rounds.drafter._round_stream
    tree = torch.empty((len(slots), 2 * NODES), dtype=torch.int32, pin_memory=True)
    with torch.cuda.stream(round_stream):
        tree.copy_(rounds.verify_tree(slots), non_blocking=True)
    round_stream.synchronize()
    assert not torch.cuda.current_stream().query()
    torch.cuda.synchronize()
    # Every verify slot holds a token (unused ones repeat the root).
    assert (tree[:, :NODES] >= 0).all()
    assert torch.equal(tree, rounds.verify_tree(slots).cpu())
    # Each tree starts at its own request's root, not a refilled row's.
    roots = rounds.drafter._pool["tokens"][:4, 0]
    assert torch.equal(tree[:, 0], roots.to(torch.int32).cpu())
    buffers.load_round_inputs(op, states, total, decode_input_ids, ngram_inputs=None)
    assert torch.equal(
        buffers.input_ids_buf[:total].cpu(), tree.flip(0)[:, :NODES].flatten()
    )
    assert torch.equal(
        buffers.seq_lens_buf[:4], states.valid_cache_lengths[order] + NODES
    )


@torch.inference_mode()
def test_a_device_chained_round_completes_without_the_host(tmp_path) -> None:
    """A round on captured graphs writes its verify tree on the GPU though
    every decision lands late: the round stream holds the tree of an eager
    round run in program order before ``complete_round`` is called."""
    _write_cost_data(str(tmp_path))
    root = str(tmp_path)
    chained = _rounds(root, late=True, forward_cycles=0, shape=_TREE)
    ordered = _rounds(root, late=False, forward_cycles=0, shape=_TREE)
    chained.capture((4, MAX_BS))
    depths = set()
    for i, bs in enumerate([1, 4, 8, 2, 6, 3]):
        trees = []
        for rounds in (chained, ordered):
            states = rounds.states
            states.future_input_map.fill_(-1)
            states.future_parent_map.fill_(-2)
            slots = rounds.draft(bs, random.Random(i))
            rounds.finish_round(bs)
            if rounds is chained:
                rounds.drafter._round_stream.synchronize()
                trees.append(rounds.verify_tree(slots))
            rounds.drafter.complete_round()
            if rounds is ordered:
                trees.append(rounds.verify_tree(slots))
        assert torch.equal(*trees)
        depths.update(max(_depths(row[NODES:])) for row in trees[1])
    assert len(depths) > 2


@torch.inference_mode()
def test_a_round_launched_behind_the_decode_graph_drafts_beside_the_forward(
    tmp_path,
) -> None:
    """``replay_decode_graph`` replays the decode graph on the round stream
    and launches its device-chained round behind it: the forward's stream
    reads what the decode graph wrote while the round still drafts, and the
    round writes the verify trees of an eager round run in program order."""
    _write_cost_data(str(tmp_path))
    root = str(tmp_path)
    # Slow depth graphs: the round outlasts the decode graph by far.
    launched = _rounds(root, late=None, forward_cycles=_LAG_CYCLES, shape=_TREE)
    ordered = _rounds(root, late=False, forward_cycles=0, shape=_TREE)
    launched.capture((4, MAX_BS))
    drafter = launched.drafter
    tail = torch.zeros((1,), dtype=torch.int64, device=drafter.device)
    decode = torch.cuda.CUDAGraph()
    with torch.cuda.graph(decode):
        torch.cuda._sleep(_LAG_CYCLES)
        tail.add_(1)
    for i, bs in enumerate([1, 4, 8, 3]):
        trees = []
        for rounds in (launched, ordered):
            slots = rounds.draft(bs, random.Random(i))
            if rounds is launched:
                drafter.prepare_round_sampling([SamplingParams(temperature=0.0)] * bs)
                drafter.replay_decode_graph(decode, bs=bs, graph_bs=rounds.graph_bs(bs))
                seen = tail.clone()
                done = torch.cuda.Event()
                done.record()
                done.synchronize()
                # The forward's stream ran past the decode graph alone.
                assert seen.item() == i + 1
                assert not drafter._round_stream.query()
            rounds.finish_round(bs)
            rounds.drafter.complete_round()
            trees.append(rounds.verify_tree(slots))
        assert torch.equal(*trees)


# The confidence rule below 3 requests, then widths 5, 2 (depth 1 still runs) and the full 16.
_CHAIN_TABLE = (ChainWidthEntry(3, 5), ChainWidthEntry(5, 2), ChainWidthEntry(7, 16))


@pytest.mark.parametrize("graphs", [False, True])
@torch.inference_mode()
def test_width_table_rounds_draft_a_chain_of_their_width(tmp_path, graphs) -> None:
    """A chain round the width table covers drafts depths 1 to ``W - 2`` (at
    least depth 1) and writes a verify tree of its ``W`` nodes: the chain the
    confidence rule drafts on the same inputs, cut or extended to ``W``, the
    window's other slots never accepted (the root, parent -1). Rounds below
    the table write the trees of a drafter without one, bit for bit; eagerly
    or replaying the captured depths."""
    root = str(tmp_path)
    _write_cost_data(root, chain=True)
    table = _rounds(root, late=None, forward_cycles=0, shape=_chain(_CHAIN_TABLE, ()))
    rule = _rounds(root, late=None, forward_cycles=0, shape=_chain((), ()))
    if graphs:
        table.capture((MAX_BS,))
    n = CHAIN_NODES
    rule_lengths, widths = set(), set()
    for i, bs in enumerate([1, 3, 5, 7, 2, 4, 6, 8, 1, 8, 3, 5, 2, 7]):
        trees, forwards = [], []
        for rounds in (table, rule):
            runner = rounds.drafter.draft_model_runner
            before = runner.forwards
            slots = rounds.draft(bs, random.Random(i))
            rounds.finish_round(bs)
            rounds.drafter.complete_round()
            trees.append(rounds.verify_tree(slots))
            forwards.append(runner.forwards - before)
        tree, rule_tree = trees
        width = chain_table_width(_CHAIN_TABLE, bs)
        widths.add(width)
        rule_length = 1 + int((rule_tree[0, n:] >= 0).sum())
        rule_lengths.add(rule_length)
        if width is None:
            assert torch.equal(tree, rule_tree)
            continue
        tokens, parents = tree[:, :n], tree[:, n:]
        chain = list(range(-1, width - 1)) + [-1] * (n - width)
        assert parents.tolist() == [chain] * bs
        common = min(width, rule_length)
        assert torch.equal(tokens[:, :common], rule_tree[:, :common])
        assert torch.equal(tokens[:, width:], tokens[:, :1].expand(-1, n - width))
        # The first step, then depths 1 to max(W - 2, 1) (replayed: none).
        assert forwards[0] == (1 if graphs else 1 + max(width - 2, 1))
    assert widths == {None, 2, 5, 16} and len(rule_lengths) > 2


@torch.inference_mode()
def test_a_width_table_round_launches_its_depths_without_waiting(tmp_path) -> None:
    """``finish_round`` launches every depth of a width table round though its
    decisions land late: none decides it. A confidence-rule round pauses
    after depth 1, whose decision has not landed."""
    root = str(tmp_path)
    _write_cost_data(root, chain=True)
    rounds = _rounds(root, late=True, forward_cycles=0, shape=_chain(_CHAIN_TABLE, ()))
    runner = rounds.drafter.draft_model_runner
    # Widths 16 (7 requests) and 5 (3 requests), then the rule (1 request).
    for bs, launched in ((7, CHAIN_STEPS - 1), (3, 3), (1, 1)):
        rounds.draft(bs, random.Random(bs))
        before = runner.forwards
        rounds.finish_round(bs)
        assert runner.forwards - before == launched
        rounds.drafter.complete_round()


@torch.inference_mode()
def test_a_padded_batch_captures_the_depths_its_rounds_reach(tmp_path) -> None:
    """Each padded batch captures the depths of the rounds that replay it
    (the live batches above the ladder entry before it): all of them where the
    confidence rule decides one, else up to its table widths' deepest
    ``W - 2`` (at least depth 1). Rounds replaying them write the trees of
    eager rounds; a round reaching a depth its padded batch did not capture is
    refused."""
    root = str(tmp_path)
    _write_cost_data(root, chain=True)
    ladder = (2, 4, 6, 8)
    floors = tuple(zip(ladder, (0, *ladder[:-1])))
    shape = _chain(_CHAIN_TABLE, floors)
    replayed = _rounds(root, late=None, forward_cycles=0, shape=shape)
    eager = _rounds(root, late=None, forward_cycles=0, shape=shape)
    replayed.capture(ladder)
    # Rule rounds of 1 and 2 requests, widths 5 (3, 4), 2 (5, 6) and 16 (7, 8).
    reached = {2: CHAIN_STEPS - 1, 4: 3, 6: 1, 8: CHAIN_STEPS - 1}
    assert set(replayed.drafter._depth_graphs) == {
        (graph_bs, depth)
        for graph_bs, deepest in reached.items()
        for depth in range(1, deepest + 1)
    }
    for i, bs in enumerate([1, 3, 5, 7, 2, 4, 6, 8]):
        trees = []
        for rounds in (replayed, eager):
            slots = rounds.draft(bs, random.Random(i))
            rounds.finish_round(bs)
            rounds.drafter.complete_round()
            trees.append(rounds.verify_tree(slots))
        assert torch.equal(*trees)
    # A rule round of one request on the width-5 graphs: they stop at depth 3.
    replayed.ladder = (4,)
    replayed.draft(1, random.Random(0))
    with pytest.raises(RuntimeError, match="did not capture"):
        replayed.finish_round(1)
    torch.cuda.synchronize()


@torch.inference_mode()
def test_rank_0_logs_its_round_counts(tmp_path, caplog) -> None:
    """With ``TOKENSPEED_EAGLE_HARNESS_ROUND_STATS=N`` the drafter logs, every
    N rounds, the cumulative rounds and requests per batch size bucket, by the
    table's width where it decides the round, else under the rule; rounds
    drafted after a prefill also count under ``prefill``."""
    root = str(tmp_path)
    _write_cost_data(root, chain=True)
    shape = _chain(_CHAIN_TABLE, ())
    with envs.TOKENSPEED_EAGLE_HARNESS_ROUND_STATS.override(3):
        rounds = _rounds(root, late=None, forward_cycles=0, shape=shape)
    with caplog.at_level(logging.INFO):
        for i, bs in enumerate([1, 3, 4, 7, 2, 8]):
            rounds.draft(bs, random.Random(i))
            rounds.finish_round(bs)
            rounds.drafter.complete_round()
    lines = [r.getMessage() for r in caplog.records]
    assert [line for line in lines if line.startswith("EAGLE-Harness rounds")] == [
        "EAGLE-Harness rounds 3: bs<=1 rule 1r/1q; bs<=4 w5 2r/7q",
        "EAGLE-Harness rounds 6: bs<=1 rule 1r/1q; bs<=2 rule 1r/2q; "
        "bs<=4 w5 2r/7q; bs<=8 w16 2r/15q",
    ]
    stats = RoundStats(2)
    assert stats.record(300, 3, True) is None
    assert stats.record(5, None, False) == (
        "EAGLE-Harness rounds 2: bs<=8 rule 1r/5q; bs<=512 prefill 1r/300q w3 1r/300q"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
