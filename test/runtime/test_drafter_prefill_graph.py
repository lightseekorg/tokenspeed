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

"""The drafter's extend-round graphs without a GPU.

Eagle records its extend round's decode steps as breakable graphs per request
count up to 16, and a split draft's K/V writer over one tile of rows. These
tests pin the plan the CUDA-graph memory projection prices, the capture order
the probe pairs its readings with, the static buffers' sizing (and the
target's hidden-row width and dtype the K/V writer takes from the prefill
graph), and which rounds replay (and when they run eager). The GPU-only steps
of a capture (warmup forwards, stream capture) are replaced by recorders;
``test_drafter_prefill_graph_cuda.py`` replays real graphs.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from test.runtime.drafter_prefill_graph_test_utils import (
    DECODE_STEPS_SERIES,
    HIDDEN,
    KV_WRITER_SERIES,
    MAX_BS,
    MAX_LEN,
    MAX_TOKENS,
    STEPS,
    VOCAB,
    WIDTH,
    RecordingObserver,
    ToyDraftModel,
    ToySplitDraftModel,
    capture_input,
    extend_input,
    make_eagle,
    prepare_extend,
)
from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

from tokenspeed.runtime.execution import model_executor  # noqa: E402
from tokenspeed.runtime.execution import (  # noqa: E402
    prefill_graph as prefill_graph_module,
)
from tokenspeed.runtime.execution.breakable_cuda_graph import run_eager  # noqa: E402
from tokenspeed.runtime.execution.cudagraph_memory import (  # noqa: E402
    estimate_cudagraph_memory,
)
from tokenspeed.runtime.execution.drafter import eagle as eagle_module  # noqa: E402
from tokenspeed.runtime.execution.drafter.eagle import Eagle  # noqa: E402
from tokenspeed.runtime.execution.forward_batch_info import (  # noqa: E402
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.execution.query_shard import QueryShardPlan  # noqa: E402
from tokenspeed.runtime.execution.tree_spec import (  # noqa: E402
    TreeSpec,
    TreeSpecConfig,
)
from tokenspeed.runtime.layers.logits_processor import (  # noqa: E402
    LogitsMetadata,
    LogitsProcessor,
)
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

register_cuda_ci(est_time=10, suite="runtime-1gpu")


# ----------------------------------------------------------------------
# The plan and the capture.
# ----------------------------------------------------------------------


def test_the_plan_takes_the_decode_steps_at_every_request_count() -> None:
    drafter, _ = make_eagle()

    # The decode steps at every request count up to the drafter's batch.
    assert drafter.extend_graph_plan([32, 16, 48]) == {
        DECODE_STEPS_SERIES: [(4, 4), (3, 3), (2, 2), (1, 1)]
    }
    # Tensor-parallel ranks run the same round, so they record alike.
    tensor_parallel, _ = make_eagle(world_size=2)
    assert tensor_parallel.extend_graph_plan([16]) == drafter.extend_graph_plan([16])
    # One draft step has no decode steps to record.
    single, _ = make_eagle(spec_num_steps=1)
    assert single.extend_graph_plan([16]) == {}
    # Its own prefill-graph hook stays undeclared: the extend graphs plan apart.
    assert not drafter.captures_prefill_graph


def test_the_decode_steps_stop_at_sixteen_requests() -> None:
    drafter, _ = make_eagle(max_bs=40)

    plan = drafter.extend_graph_plan([16])

    assert plan[DECODE_STEPS_SERIES] == [(bs, bs) for bs in range(16, 0, -1)]


@pytest.mark.parametrize(
    "setup",
    [
        "attention_dp",
        "pipeline",
        "shared_mtp_topk",
        "draft_tree",
        "speculative_sampling",
        "draft_token_history",
    ],
)
def test_configurations_the_graphs_cannot_bake_record_nothing(setup) -> None:
    model = ToyDraftModel("cpu")
    if setup == "shared_mtp_topk":
        model.index_share_for_mtp_iteration = True
    drafter, _ = make_eagle(
        model=model,
        dp_size=2 if setup == "attention_dp" else 1,
        world_size=2 if setup == "pipeline" else 1,
        pp_size=2 if setup == "pipeline" else 1,
    )
    # The state bind_tree, bind_sampling_backend and a draft model that reads
    # request-token history leave on the drafter.
    if setup == "draft_tree":
        drafter.tree_spec = TreeSpec(
            TreeSpecConfig(topk=2, num_steps=STEPS, num_nodes=2 * STEPS + 1),
            MAX_BS,
            "cpu",
        )
    if setup == "speculative_sampling":
        drafter.draft_sampler = object()
    if setup == "draft_token_history":
        drafter.draft_reads_token_history = True
    assert drafter.extend_graph_plan([16, 32]) == {}


def _capture_without_a_gpu(monkeypatch, drafter, capture, observer):
    """Run capture_extend_graphs with the GPU-only steps (warmup forwards,
    stream capture) replaced by recorders."""
    events = []

    def decode_steps(self, bs, capture, window):
        with window:
            events.append(("decode steps", bs))
        return eagle_module._ExtendGraph(capture=None, ctx=None)

    def kv_writer(self, rows, capture, window):
        with window:
            events.append(("kv writer", rows))

    monkeypatch.setattr(Eagle, "_capture_decode_steps", decode_steps)
    monkeypatch.setattr(Eagle, "_capture_kv_tile", kv_writer)
    monkeypatch.setattr(
        Eagle,
        "_run_first_step_eager",
        lambda self, bs, stream: events.append(("eager step 0", bs)),
    )
    drafter.capture_extend_graphs(capture, observer=observer)
    return events


def test_the_capture_records_the_sampled_entries_in_ladder_order(monkeypatch) -> None:
    """The capture takes the sampled counts widest first, each after step 0."""
    drafter, backend = make_eagle()
    prepared = []
    capture = capture_input(drafter, backend, [16, 32], entries=3, prepared=prepared)
    observer = RecordingObserver()

    events = _capture_without_a_gpu(monkeypatch, drafter, capture, observer)

    # A probe of three entries over four counts takes the widest three.
    assert events == [
        event
        for bs in (4, 3, 2)
        for event in (("eager step 0", bs), ("decode steps", bs))
    ]
    assert prepared == [(4, 4), (3, 3), (2, 2)]
    assert observer.order == [DECODE_STEPS_SERIES] * 3
    # The projection accepts exactly these readings for these ladders.
    estimate = estimate_cudagraph_memory(observer.samples, capture.ladders)
    assert estimate.measured_total == 3 << 20


def test_a_split_draft_captures_its_steps_then_its_kv_writer(monkeypatch) -> None:
    """A split draft's steps open at step 0, so no eager step 0 runs before them."""
    drafter, backend = make_eagle(model=ToySplitDraftModel("cpu"))
    capture = capture_input(drafter, backend, [16, 32], entries=None)
    observer = RecordingObserver()

    events = _capture_without_a_gpu(monkeypatch, drafter, capture, observer)

    assert events == [
        *[("decode steps", bs) for bs in (4, 3, 2, 1)],
        ("kv writer", 32),
    ]
    assert observer.order == [DECODE_STEPS_SERIES] * 4 + [KV_WRITER_SERIES]
    # The steps read a valid token and a finite hidden row per request.
    buffers = drafter._extend_buffers
    assert torch.equal(buffers.draft_ids, torch.ones_like(buffers.draft_ids))
    assert not buffers.hidden_out.any()


def test_static_buffers_fit_the_whole_plan_and_survive_a_recapture(
    monkeypatch,
) -> None:
    """A probe sizes the buffers for the plan; only a new hidden width reallocates."""
    drafter, backend = make_eagle()

    probe = capture_input(drafter, backend, [16, 32], entries=3)
    _capture_without_a_gpu(monkeypatch, drafter, probe, RecordingObserver())

    # The decode steps' request counts reach the drafter's batch.
    buffers = drafter._extend_buffers
    assert buffers.hidden_in.shape == (MAX_BS, WIDTH)
    assert buffers.hidden_in.dtype == torch.float32
    assert buffers.gather_ids.shape == (MAX_BS,)
    assert buffers.frontier.shape == (MAX_BS,)
    assert buffers.frontier.dtype == drafter.input_buffers.seq_lens_buf.dtype
    assert buffers.next_tokens.shape == (MAX_BS, STEPS + 1)

    full = capture_input(drafter, backend, [16, 32], entries=None)
    _capture_without_a_gpu(monkeypatch, drafter, full, RecordingObserver())
    assert drafter._extend_buffers is buffers
    # Every planned entry was captured this time.
    assert sorted(drafter._decode_steps_graphs) == [1, 2, 3, 4]

    wider = capture_input(drafter, backend, [16, 32], entries=None, width=64)
    _capture_without_a_gpu(monkeypatch, drafter, wider, RecordingObserver())
    assert drafter._extend_buffers is not buffers
    assert drafter._extend_buffers.hidden_in.shape == (MAX_BS, 64)


def test_a_capture_without_warmup_is_refused() -> None:
    drafter, backend = make_eagle()
    capture = dataclasses.replace(
        capture_input(drafter, backend, [16], entries=None),
        num_warmup=0,
    )

    with pytest.raises(ValueError, match="at least one warmup"):
        drafter.capture_extend_graphs(capture, observer=RecordingObserver())


def test_a_split_draft_plans_its_kv_writer_beside_the_decode_steps() -> None:
    drafter, _ = make_eagle(model=ToySplitDraftModel("cpu"))

    # One tile no wider than the largest bucket.
    assert drafter.extend_graph_plan([16, 32]) == {
        KV_WRITER_SERIES: [(32, 1)],
        DECODE_STEPS_SERIES: [(4, 4), (3, 3), (2, 2), (1, 1)],
    }
    wide = drafter.extend_graph_plan([16, 8192])
    assert wide[KV_WRITER_SERIES] == [(2048, 1)]
    # Attention DP keeps step 0 whole: nothing is recorded.
    dp, _ = make_eagle(model=ToySplitDraftModel("cpu"), dp_size=2, world_size=2)
    assert dp.extend_graph_plan([16, 32]) == {}


@pytest.mark.parametrize("final", [True, False])
def test_a_split_round_drafts_what_the_narrowed_round_drafts(final) -> None:
    """A split round leaves the narrowed round's window and cache (a mid chunk none)."""
    lengths, prefixes = [9, 4, 1], [0, 20, 5]
    generator = torch.Generator().manual_seed(1)
    num_tokens = sum(lengths)
    ids = torch.randint(0, VOCAB, (num_tokens,), generator=generator)
    sampled = torch.randint(0, VOCAB, (len(lengths),), generator=generator).int()
    hidden = torch.randn(num_tokens, WIDTH, generator=generator)
    rounds = []
    for model in (ToyDraftModel("cpu"), ToySplitDraftModel("cpu")):
        drafter, backend = make_eagle(model=model)
        prepare_extend(drafter, backend, lengths, prefixes, final=final, ids=ids)
        window = drafter.draft(
            extend_input(drafter, lengths, sampled=sampled, hidden=hidden)
        )
        rounds.append((window, backend.cache, drafter.draft_model_runner.forward_calls))
    (narrowed, narrowed_cache, narrowed_calls), (split, split_cache, split_calls) = (
        rounds
    )

    assert torch.equal(split_cache[1:], narrowed_cache[1:])
    if final:
        assert torch.equal(split, narrowed)
        assert (narrowed_calls, split_calls) == (STEPS, STEPS)
    else:
        assert torch.equal(split[:, 0], narrowed[:, 0])
        assert torch.equal(split, split[:, :1].expand_as(split))
        assert (narrowed_calls, split_calls) == (1, 0)


# ----------------------------------------------------------------------
# The target's hidden rows the K/V writer reads.
# ----------------------------------------------------------------------


def _target_prefill_graph() -> prefill_graph_module.PrefillGraph:
    """The target's prefill graph over token buckets 16 and 32, built as the
    executor builds it; the model runner stands in for a loaded model."""
    model_runner = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(embed_tokens=object())),
        model_config=SimpleNamespace(requires_request_token_history=False),
        is_generation=True,
        is_multimodal=False,
    )
    config = model_executor.ModelExecutorConfig(
        max_req_pool_size=MAX_BS + 1,
        output_length=STEPS + 1,
        enforce_eager=False,
        prefix_granularity=128,
        max_num_seqs=MAX_BS,
        chunked_prefill_size=MAX_TOKENS,
        input_vocab_size=VOCAB,
        vocab_size=VOCAB,
        context_len=MAX_LEN,
        physical_context_len=MAX_LEN,
        device="cpu",
        gpu_id=0,
        global_rank=0,
        cudagraph_capture_sizes=None,
        disable_cuda_graph_padding=False,
        spec_topk=1,
        max_cudagraph_capture_size=MAX_BS,
        model_is_mrope=False,
        autotune_cache_key=None,
        prefill_only=False,
        decode_only_attention=False,
        prefill_graph_capture_batch_sizes=None,
        input_logprob_chunk_tokens=1024,
        enable_speculative_sampling=False,
        query_shard_size=1,
        query_shard_rank=0,
        enable_memory_saver=False,
        prefill_graph_max_tokens=32,
        prefill_graph_capture_sizes=[16, 32],
    )
    return prefill_graph_module.PrefillGraph(
        model_runner=model_runner,
        attn_backend=None,
        token_to_kv_pool=None,
        input_buffers=None,
        config=config,
        memory_saver_adapter=TorchMemorySaverAdapter.create(enable=False),
    )


@pytest.mark.parametrize("taps", [0, 1, 3])
def test_the_prefill_graph_reports_the_rows_a_full_capture_stores(taps) -> None:
    """The drafter's static buffer for the target's hidden rows takes its
    width and dtype from ``PrefillGraph.captured_hidden_spec``, and a replay
    refuses rows of another width or dtype: the spec must be what the logits
    processor's FULL capture stores from the graph's outputs, the aux taps
    side by side, else the final hidden rows. ``taps=0`` is a target without
    taps (an MTP target); a tap is wider than the final hidden rows here, so
    the two widths are told apart."""
    graph = _target_prefill_graph()
    with pytest.raises(RuntimeError, match="captured no output rows"):
        graph.captured_hidden_spec()
    rows = 32
    hidden = torch.randn(rows, HIDDEN, dtype=torch.bfloat16)
    aux = [torch.randn(rows, WIDTH, dtype=torch.bfloat16) for _ in range(taps)]
    # The spec a capture's warmups read off the inner forward's outputs.
    outputs = prefill_graph_module.CapturedForward(hidden, aux or None)
    graph._reserve_outputs(prefill_graph_module._output_spec(outputs))

    processor = LogitsProcessor(
        attn_tp_size=1, config=PretrainedConfig(vocab_size=VOCAB), dp_lm_head_tp=False
    )
    stored = processor(
        input_ids=None,
        hidden_states=hidden,
        lm_head=torch.nn.Linear(HIDDEN, VOCAB, bias=False, dtype=torch.bfloat16),
        logits_metadata=LogitsMetadata(
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            gather_ids=torch.tensor([rows - 1]),
            query_shard=None,
        ),
        aux_hidden_states=aux or None,
    ).hidden_states

    # The taps side by side, else the final hidden rows.
    assert (stored.shape[1], stored.dtype) == (
        WIDTH * taps if taps else HIDDEN,
        torch.bfloat16,
    )
    assert graph.captured_hidden_spec() == (stored.shape[1], stored.dtype)


# ----------------------------------------------------------------------
# Which graphs a round replays.
# ----------------------------------------------------------------------


def _captured(monkeypatch, buckets, *, spec_num_steps=STEPS, model=None):
    drafter, backend = make_eagle(spec_num_steps=spec_num_steps, model=model)
    capture = capture_input(drafter, backend, buckets, entries=None)
    _capture_without_a_gpu(monkeypatch, drafter, capture, RecordingObserver())
    return drafter


def test_a_pure_extend_replays_the_decode_steps_at_its_request_count(
    monkeypatch,
) -> None:
    drafter = _captured(monkeypatch, [16, 32, 64])

    # Any token count: past the largest bucket too.
    for lengths in ([1], [16], [17], [40, 24], [65], [40, 40], [1, 1, 1], [3] * 4):
        assert drafter._replays_decode_steps(
            len(lengths), extend_input(drafter, lengths)
        ), lengths
    # Hidden rows not one per input row: step 0 runs eager either way.
    short = extend_input(drafter, [8], hidden=torch.randn(1, WIDTH))
    assert drafter._replays_decode_steps(1, short)
    # One draft step has no decode steps: eager.
    single = _captured(monkeypatch, [16], spec_num_steps=1)
    assert not single._replays_decode_steps(3, extend_input(single, [1, 1, 1]))


def test_a_count_past_the_decode_steps_ladder_runs_them_eager(monkeypatch) -> None:
    monkeypatch.setattr(eagle_module, "EXTEND_DECODE_MAX_REQUESTS", 2)
    drafter = _captured(monkeypatch, [16])

    assert sorted(drafter._decode_steps_graphs) == [1, 2]
    assert not drafter._replays_decode_steps(3, extend_input(drafter, [2, 2, 2]))


def test_rounds_a_graph_did_not_bake_run_eager(monkeypatch) -> None:
    drafter = _captured(monkeypatch, [16, 32])

    # A MIXED round (one prefill, one decode request).
    mixed = extend_input(drafter, [5, 4])
    mixed.num_extends = 1
    mixed.forward_mode = ForwardMode.MIXED
    assert not drafter._replays_decode_steps(2, mixed)
    # Pure decode.
    decode = extend_input(drafter, [4])
    decode.num_extends = 0
    decode.forward_mode = ForwardMode.DECODE
    assert not drafter._replays_decode_steps(1, decode)
    # Attention-DP sizing.
    dp = extend_input(drafter, [5], global_num_tokens=[5, 9], global_bs=[1, 1])
    assert not drafter._replays_decode_steps(1, dp)
    # A query shard of the span.
    shard = QueryShardPlan.from_forward(
        total_tokens=8,
        input_lengths=[8],
        size=2,
        rank=0,
        device=drafter.input_buffers.seq_lens_buf.device,
    )
    sharded = extend_input(drafter, [8], query_shard=shard)
    assert not drafter._replays_decode_steps(1, sharded)
    # Nothing captured (or released).
    drafter.release_extend_graphs()
    assert not drafter._replays_decode_steps(1, extend_input(drafter, [8]))
    # A split draft's steps open at step 0: only a split round replays them.
    split = _captured(monkeypatch, [16, 32], model=ToySplitDraftModel("cpu"))
    assert sorted(split._decode_steps_graphs) == [1, 2, 3, 4]
    assert not split._replays_decode_steps(1, extend_input(split, [8]))


def test_run_eager_is_a_direct_call_outside_a_capture() -> None:
    calls = []
    assert run_eager(lambda: calls.append("ran") or 7) == 7
    assert calls == ["ran"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
