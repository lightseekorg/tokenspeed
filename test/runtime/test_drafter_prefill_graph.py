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

"""The drafter prefill graph (``--enable-drafter-prefill-graph``) without a GPU.

Eagle records its extend round as breakable graphs: step 0 per (prefill token
bucket, request count), the decode steps per request count. These tests pin
the plan the CUDA-graph memory projection prices, the capture order the probe
pairs its readings with, the static buffers' sizing (and the target's hidden-row
width and dtype it takes from the prefill graph), which graphs a round replays
(and when it runs eager), and the launches the switch refuses. The
GPU-only steps of a capture (warmup forwards, stream capture) are replaced by
recorders; ``test_drafter_prefill_graph_cuda.py`` replays real graphs.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from test.runtime.drafter_prefill_graph_test_utils import (
    DECODE_STEPS_SERIES,
    FIRST_STEP_SERIES,
    HIDDEN,
    MAX_BS,
    MAX_LEN,
    MAX_TOKENS,
    STEPS,
    VOCAB,
    WIDTH,
    RecordingObserver,
    ToyDraftModel,
    capture_input,
    extend_input,
    make_eagle,
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
from tokenspeed.runtime.utils.server_args import ServerArgs  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")


# ----------------------------------------------------------------------
# The plan and the capture.
# ----------------------------------------------------------------------


def test_the_plan_takes_the_target_buckets_at_their_request_counts() -> None:
    drafter, _ = make_eagle()

    plan = drafter.extend_graph_plan(
        [32, 16, 48], lambda bucket: [1, 2] if bucket >= 32 else [1]
    )

    assert plan == {
        FIRST_STEP_SERIES: [(48, 1), (48, 2), (32, 1), (32, 2), (16, 1)],
        DECODE_STEPS_SERIES: [(2, 2), (1, 1)],
    }
    # One draft step has no decode steps to record.
    single, _ = make_eagle(spec_num_steps=1)
    assert single.extend_graph_plan([16], lambda _bucket: [1]) == {
        FIRST_STEP_SERIES: [(16, 1)]
    }
    # Its own prefill-graph hook stays undeclared: the extend graphs are a
    # separate plan, captured only behind --enable-drafter-prefill-graph.
    assert not drafter.captures_prefill_graph


@pytest.mark.parametrize(
    "setup",
    [
        "attention_dp",
        "tensor_parallel",
        "shared_mtp_topk",
        "undeclared_draft",
        "draft_tree",
        "speculative_sampling",
        "draft_token_history",
    ],
)
def test_configurations_the_graphs_cannot_bake_record_nothing(setup) -> None:
    model = ToyDraftModel("cpu")
    if setup == "shared_mtp_topk":
        model.index_share_for_mtp_iteration = True
    if setup == "undeclared_draft":
        # A draft that does not declare its narrowed step breakable (an MLA
        # draft attends inside one unbroken call).
        model.supports_breakable_draft_extend = False
    drafter, _ = make_eagle(
        model=model,
        dp_size=2 if setup == "attention_dp" else 1,
        world_size=2 if setup == "tensor_parallel" else 1,
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
    assert drafter.extend_graph_plan([16, 32], lambda _bucket: [1]) == {}


def _capture_without_a_gpu(monkeypatch, drafter, capture, observer):
    """Run capture_extend_graphs with the GPU-only steps (warmup forwards,
    stream capture) replaced by recorders."""
    events = []

    def first_step(self, bucket, bs, capture, window):
        with window:
            events.append(("first step", bucket, bs))
        return eagle_module._ExtendGraph(capture=None, ctx=None)

    def decode_steps(self, bs, capture, window):
        with window:
            events.append(("decode steps", bs))
        return eagle_module._ExtendGraph(capture=None, ctx=None)

    monkeypatch.setattr(Eagle, "_capture_first_step", first_step)
    monkeypatch.setattr(Eagle, "_capture_decode_steps", decode_steps)
    monkeypatch.setattr(
        Eagle,
        "_run_first_step_eager",
        lambda self, bucket, bs, stream: events.append(("eager step 0", bucket, bs)),
    )
    drafter.capture_extend_graphs(capture, observer=observer)
    return events


def test_the_capture_records_the_sampled_entries_in_ladder_order(monkeypatch) -> None:
    """The probe pairs each reading with a ladder position in order, so the
    capture must take the sampled entries widest first, the decode steps
    after every step-0 graph and each behind a step 0 at its request count."""
    drafter, backend = make_eagle()
    buckets = [16, 32, 48, 64, 80, 96, 112]
    prepared = []
    capture = capture_input(
        drafter,
        backend,
        buckets,
        lambda bucket: [1, 2, 3] if bucket >= 48 else [1, 2],
        entries=5,
        prepared=prepared,
    )
    observer = RecordingObserver()

    events = _capture_without_a_gpu(monkeypatch, drafter, capture, observer)

    # Widths 112, 96, 80 and 48 are the probe's positions 0, 1, 2 and 4.
    first = [(b, n) for b in (112, 96, 80, 48) for n in (1, 2, 3)]
    assert events == [
        *[("first step", b, n) for b, n in first],
        ("eager step 0", 48, 3),
        ("decode steps", 3),
        ("eager step 0", 16, 2),
        ("decode steps", 2),
        ("eager step 0", 16, 1),
        ("decode steps", 1),
    ]
    assert prepared == [*first, (48, 3), (16, 2), (16, 1)]
    assert (
        observer.order == [FIRST_STEP_SERIES] * len(first) + [DECODE_STEPS_SERIES] * 3
    )
    # The projection accepts exactly these readings for these ladders.
    estimate = estimate_cudagraph_memory(observer.samples, capture.ladders)
    assert estimate.measured_total == (len(first) + 3) << 20


def test_static_buffers_fit_the_whole_plan_and_survive_a_recapture(
    monkeypatch,
) -> None:
    """The static buffers are sized once for the plan's largest bucket and
    request count, by a probe that samples a few entries too, and a later
    capture of the same plan (after a cache rebind) keeps them; one whose
    target hidden rows differ in width allocates new ones."""
    drafter, backend = make_eagle()
    buckets = [16, 32, 64, 128]

    def counts(bucket):
        return [1, 2, 3] if bucket >= 64 else [1]

    probe = capture_input(drafter, backend, buckets, counts, entries=2)
    _capture_without_a_gpu(monkeypatch, drafter, probe, RecordingObserver())

    buffers = drafter._extend_buffers
    assert buffers.hidden_in.shape == (128, WIDTH)
    assert buffers.hidden_in.dtype == torch.float32
    assert buffers.gather_ids.shape == (3,)
    assert buffers.frontier.shape == (3,)
    assert buffers.frontier.dtype == drafter.input_buffers.seq_lens_buf.dtype
    assert buffers.next_tokens.shape == (3, STEPS + 1)

    full = capture_input(drafter, backend, buckets, counts, entries=None)
    _capture_without_a_gpu(monkeypatch, drafter, full, RecordingObserver())
    assert drafter._extend_buffers is buffers
    # Every planned entry was captured this time.
    assert sorted(drafter._first_step_graphs) == sorted(full.plan[FIRST_STEP_SERIES])

    wider = capture_input(drafter, backend, buckets, counts, entries=None, width=64)
    _capture_without_a_gpu(monkeypatch, drafter, wider, RecordingObserver())
    assert drafter._extend_buffers is not buffers
    assert drafter._extend_buffers.hidden_in.shape == (128, 64)


def test_a_capture_without_warmup_is_refused() -> None:
    drafter, backend = make_eagle()
    capture = dataclasses.replace(
        capture_input(drafter, backend, [16], lambda _bucket: [1], entries=None),
        num_warmup=0,
    )

    with pytest.raises(ValueError, match="at least one warmup"):
        drafter.capture_extend_graphs(capture, observer=RecordingObserver())


# ----------------------------------------------------------------------
# The target's hidden rows the step-0 graph reads.
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
        enable_drafter_prefill_graph=True,
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
        config=PretrainedConfig(vocab_size=VOCAB), dp_lm_head_tp=False
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


def _captured(monkeypatch, buckets, counts, *, spec_num_steps=STEPS):
    drafter, backend = make_eagle(spec_num_steps=spec_num_steps)
    capture = capture_input(drafter, backend, buckets, counts, entries=None)
    _capture_without_a_gpu(monkeypatch, drafter, capture, RecordingObserver())
    return drafter


def test_a_pure_extend_replays_the_smallest_fitting_bucket(monkeypatch) -> None:
    drafter = _captured(
        monkeypatch, [16, 32, 64], lambda bucket: [1, 2] if bucket >= 32 else [1]
    )
    route = eagle_module._ExtendRoute
    cases = {
        # (lengths) -> route
        (1,): route((16, 1), True),
        (16,): route((16, 1), True),
        (17,): route((32, 1), True),
        (40, 24): route((64, 2), True),
        (10, 7): route((32, 2), True),
        # Over the largest bucket: step 0 eager, the decode steps replay.
        (65,): route(None, True),
        (40, 40): route(None, True),
    }
    for lengths, expected in cases.items():
        assert (
            drafter._extend_graph_route(
                len(lengths), extend_input(drafter, list(lengths))
            )
            == expected
        ), lengths
    # No graph at this request count at all: eager.
    assert drafter._extend_graph_route(3, extend_input(drafter, [1, 1, 1])) is None


def test_rounds_a_graph_did_not_bake_run_eager(monkeypatch) -> None:
    drafter = _captured(monkeypatch, [16, 32], lambda _bucket: [1, 2])

    # A MIXED round (one prefill, one decode request).
    mixed = extend_input(drafter, [5, 4])
    mixed.num_extends = 1
    mixed.forward_mode = ForwardMode.MIXED
    assert drafter._extend_graph_route(2, mixed) is None
    # Pure decode.
    decode = extend_input(drafter, [4])
    decode.num_extends = 0
    decode.forward_mode = ForwardMode.DECODE
    assert drafter._extend_graph_route(1, decode) is None
    # Attention-DP sizing.
    dp = extend_input(drafter, [5], global_num_tokens=[5, 9], global_bs=[1, 1])
    assert drafter._extend_graph_route(1, dp) is None
    # A query shard of the span.
    shard = QueryShardPlan.from_forward(
        total_tokens=8,
        input_lengths=[8],
        size=2,
        rank=0,
        device=drafter.input_buffers.seq_lens_buf.device,
    )
    sharded = extend_input(drafter, [8], query_shard=shard)
    assert drafter._extend_graph_route(1, sharded) is None
    # Target hidden rows that are not one per input row.
    short = extend_input(drafter, [8], hidden=torch.randn(1, WIDTH))
    assert drafter._extend_graph_route(1, short) is None
    # Nothing captured (or released).
    drafter.release_extend_graphs()
    assert drafter._extend_graph_route(1, extend_input(drafter, [8])) is None


def test_a_replay_refuses_target_hidden_rows_it_did_not_record(monkeypatch) -> None:
    """The graphs read the target's hidden rows from a static buffer of the
    captured width and dtype; a round with other rows is refused before
    anything replays."""
    drafter = _captured(monkeypatch, [16], lambda _bucket: [1])

    with pytest.raises(RuntimeError, match="not the ones the drafter prefill"):
        drafter.draft(extend_input(drafter, [8], hidden=torch.randn(8, WIDTH + 8)))
    with pytest.raises(RuntimeError, match="not the ones the drafter prefill"):
        drafter.draft(
            extend_input(
                drafter, [8], hidden=torch.randn(8, WIDTH, dtype=torch.bfloat16)
            )
        )


def test_run_eager_is_a_direct_call_outside_a_capture() -> None:
    calls = []
    assert run_eager(lambda: calls.append("ran") or 7) == 7
    assert calls == ["ran"]


# ----------------------------------------------------------------------
# The launches the switch refuses.
# ----------------------------------------------------------------------


def _args(**overrides) -> ServerArgs:
    kwargs = dict(
        model="x",
        speculative_algorithm="MTP",
        enable_drafter_prefill_graph=True,
    )
    kwargs.update(overrides)
    return ServerArgs(**kwargs)


def test_the_switch_is_off_by_default_and_parses() -> None:
    assert ServerArgs(model="x").enable_drafter_prefill_graph is False
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    assert parser.parse_args(["--model", "x"]).enable_drafter_prefill_graph is False
    assert parser.parse_args(
        ["--model", "x", "--enable-drafter-prefill-graph"]
    ).enable_drafter_prefill_graph


@pytest.mark.parametrize("algorithm", ["MTP", "EAGLE3"])
def test_a_single_rank_chain_drafter_resolves(algorithm) -> None:
    extra = {} if algorithm == "MTP" else {"speculative_draft_model_path": "draft"}
    assert _args(speculative_algorithm=algorithm, **extra).enable_drafter_prefill_graph


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"speculative_algorithm": None}, "needs a chain drafter"),
        (
            {"speculative_algorithm": "DFLASH", "speculative_draft_model_path": "d"},
            "needs a chain drafter",
        ),
        (
            {
                "speculative_eagle_topk": 2,
                "speculative_num_steps": 3,
                "speculative_num_draft_tokens": 6,
            },
            "draft trees",
        ),
        (
            {"enable_speculative_sampling": True, "sampling_backend": "flashinfer"},
            "speculative-sampling",
        ),
        ({"attn_tp_size": 2}, "runs on one rank"),
        ({"disable_prefill_graph": True}, "prefill CUDA graphs"),
        ({"enforce_eager": True}, "prefill CUDA graphs"),
    ],
)
def test_launches_the_graphs_cannot_serve_are_refused(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        _args(**overrides)


def _executor(*, enabled, prefill_graph_disabled, drafter):
    """The executor state the boot-time verdict reads, without the GPU boot."""
    executor = model_executor.ModelExecutor.__new__(model_executor.ModelExecutor)
    executor.config = SimpleNamespace(
        enable_drafter_prefill_graph=enabled,
        prefill_graph_capture_batch_sizes=None,
        context_len=4096,
        max_num_seqs=MAX_BS,
        data_parallel_size=1,
    )
    executor.prefill_graph = SimpleNamespace(
        disable=prefill_graph_disabled, capture_buckets=[16, 32]
    )
    executor.drafter = drafter
    return executor


def test_the_executor_refuses_a_boot_that_would_record_no_drafter_graph() -> None:
    supported, _ = make_eagle()
    model = ToyDraftModel("cpu")
    model.supports_breakable_draft_extend = False
    undeclared, _ = make_eagle(model=model)

    _executor(
        enabled=True, prefill_graph_disabled=False, drafter=supported
    )._validate_drafter_extend_graphs()
    with pytest.raises(ValueError, match="prefill CUDA graph is off"):
        _executor(
            enabled=True, prefill_graph_disabled=True, drafter=supported
        )._validate_drafter_extend_graphs()
    with pytest.raises(ValueError, match="Eagle drafter records none"):
        _executor(
            enabled=True, prefill_graph_disabled=False, drafter=undeclared
        )._validate_drafter_extend_graphs()
    with pytest.raises(ValueError, match="no drafter"):
        _executor(
            enabled=True, prefill_graph_disabled=False, drafter=None
        )._validate_drafter_extend_graphs()
    # Without the switch nothing is required.
    for drafter in (supported, undeclared, None):
        executor = _executor(
            enabled=False, prefill_graph_disabled=True, drafter=drafter
        )
        executor._validate_drafter_extend_graphs()
        assert not executor.captures_drafter_extend_graphs
