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

"""Drafter prefill graph replays on a GPU (``--enable-drafter-prefill-graph``).

A real Eagle drafter over the toy draft model and backend
(``drafter_prefill_graph_test_utils``) captures its extend-round graphs; each
round then replays them and must leave the same draft tokens and draft cache
as a second drafter running the same round eager. Also covered: a step
published through ``run_eager`` re-runs on every replay with the host state
of that moment, recording such a publication into the graph instead makes the
replay diverge, and a replay waits for a Host load into the draft cache.

The last test serves a real Qwen4-Exp MTP checkpoint with and without the
switch; it runs only when ``TOKENSPEED_DRAFTER_PREFILL_GRAPH_MODEL`` names the
checkpoint (``TOKENSPEED_DRAFTER_PREFILL_GRAPH_ARGS`` adds engine flags), for
example::

    TOKENSPEED_DRAFTER_PREFILL_GRAPH_MODEL=nvidia/Qwen3.8-Flash-Next-NVFP4 \\
    TOKENSPEED_DRAFTER_PREFILL_GRAPH_ARGS="--kv-cache-dtype fp8_e4m3" \\
    python3 -m pytest -s test/runtime/test_drafter_prefill_graph_cuda.py
"""

from __future__ import annotations

import os
import random
import re
import shlex
import sys
from test.runtime.drafter_prefill_graph_test_utils import (
    STEPS,
    VOCAB,
    WIDTH,
    ToyBackend,
    ToyDraftModel,
    capture_input,
    extend_input,
    make_eagle,
    prepare_extend,
)

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

from tokenspeed.runtime.cache.l2.layerwise_load import (  # noqa: E402
    LayerwiseLoadTracker,
)
from tokenspeed.runtime.execution.breakable_cuda_graph import (  # noqa: E402
    BreakableCapture,
    run_eager,
)
from tokenspeed.runtime.execution.drafter import eagle as eagle_module  # noqa: E402
from tokenspeed.runtime.execution.drafter.eagle import Eagle  # noqa: E402
from tokenspeed.runtime.execution.memory_delta import (  # noqa: E402
    NULL_MEMORY_DELTA_OBSERVER,
)

register_cuda_ci(est_time=60, suite="runtime-1gpu")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


def test_run_eager_reruns_on_replay_with_live_host_state() -> None:
    """A break recorded by run_eager re-runs on every replay and reads the
    host state of that moment: here, which buffer a captured segment then
    adds is chosen by host state the step itself reads."""
    x = torch.zeros(4, device="cuda")
    buffers = [torch.zeros(4, device="cuda") for _ in range(3)]
    chosen = {"index": 0}
    published = torch.zeros(4, device="cuda")

    def publish() -> None:
        published.copy_(buffers[chosen["index"]])

    cap = BreakableCapture()
    with torch.inference_mode():
        with cap:
            x.add_(1)
            run_eager(publish)
            x.add_(published)
        for index in (1, 2, 0):
            buffers[index].fill_(10.0 * (index + 1))
            chosen["index"] = index
            before = x.clone()
            cap.replay()
            torch.cuda.synchronize()
            assert torch.equal(x, before + 1 + 10.0 * (index + 1))


def _capture(drafter: Eagle, backend: ToyBackend, buckets, counts) -> None:
    capture = capture_input(
        drafter, backend, buckets, counts, entries=None, stream=torch.cuda.Stream()
    )
    with torch.inference_mode():
        drafter.capture_extend_graphs(capture, observer=NULL_MEMORY_DELTA_OBSERVER)
    torch.cuda.synchronize()
    # Start the rounds from a clean cache.
    backend.cache.zero_()


def _round(drafter, backend, lengths, prefixes, final, inputs):
    """Lay out the round on ``drafter`` and draft it; returns the verify
    window and the eager draft forwards the round ran."""
    ids, sampled, hidden = inputs
    prepare_extend(drafter, backend, lengths, prefixes, final=final, ids=ids)
    runner = drafter.draft_model_runner
    before = runner.forward_calls
    window = drafter.draft(
        extend_input(drafter, lengths, sampled=sampled, hidden=hidden)
    )
    torch.cuda.synchronize()
    return window.clone(), runner.forward_calls - before


def _inputs(lengths, generator):
    num_tokens = sum(lengths)
    return (
        torch.randint(0, VOCAB, (num_tokens,), generator=generator),
        torch.randint(0, VOCAB, (len(lengths),), generator=generator).int(),
        torch.randn(num_tokens, WIDTH, generator=generator),
    )


def test_replays_match_the_eager_round() -> None:
    """Exact and padded buckets, one to three requests, cached prefixes,
    final chunks (the sampled token patched into the last row) and mid chunks
    (step 0 alone, its cache writes kept), and the decode steps replaying
    behind an eager step 0 past the largest bucket and at a request count no
    step-0 graph takes: the replayed round leaves the same verify window and
    draft cache as the eager one, the dummy slot aside. The toy's
    token-shaped part is row-wise, so padded buckets match exactly too."""
    generator = torch.Generator().manual_seed(0)
    graphed, graphed_backend = make_eagle(device="cuda")
    eager, eager_backend = make_eagle(device="cuda")
    _capture(
        graphed,
        graphed_backend,
        [16, 32, 64],
        lambda bucket: [1, 2, 3] if bucket >= 32 else [1, 2],
    )
    route = eagle_module._ExtendRoute
    cases = [
        # (lengths, prefixes, final, expected route)
        ([16], [0], True, route((16, 1), True)),
        ([5], [40], True, route((16, 1), True)),
        ([9, 7], [0, 100], True, route((16, 2), True)),
        ([20, 3, 1], [7, 0, 300], True, route((32, 3), True)),
        ([30], [0], False, route((32, 1), True)),
        ([12, 12], [64, 0], False, route((32, 2), True)),
        ([64], [128], True, route((64, 1), True)),
        ([33, 31], [0, 0], True, route((64, 2), True)),
        ([70], [0], True, route(None, True)),
        ([3], [0], True, route((16, 1), True)),
        # Four requests: no step-0 graph at this request count.
        ([3, 3, 3, 3], [0, 0, 0, 0], True, route(None, True)),
    ]
    with torch.inference_mode():
        for lengths, prefixes, final, expected in cases:
            inputs = _inputs(lengths, generator)
            prepare_extend(
                graphed, graphed_backend, lengths, prefixes, final=final, ids=inputs[0]
            )
            assert (
                graphed._extend_graph_route(
                    len(lengths), extend_input(graphed, lengths, hidden=inputs[2])
                )
                == expected
            ), lengths
            window, forwards = _round(
                graphed, graphed_backend, lengths, prefixes, final, inputs
            )
            want, eager_forwards = _round(
                eager, eager_backend, lengths, prefixes, final, inputs
            )
            # A replayed step runs no eager forward.
            if expected is None:
                assert forwards == eager_forwards
            else:
                assert forwards == (0 if expected.first_step is not None else 1)
            assert eager_forwards == (STEPS if final else 1)
            assert torch.equal(window, want), lengths
            assert torch.equal(graphed_backend.cache[1:], eager_backend.cache[1:])
            if not final:
                # A mid chunk drafts step 0 only; the rest repeat column 0.
                assert torch.equal(window[:, 2], window[:, 0])


def test_the_decode_steps_replay_without_a_step_0_graph() -> None:
    """A draft model whose narrowed step 0 has no breaks (the configuration
    every tensor-parallel launch takes too) records the decode steps alone:
    every round runs step 0 eager and replays the decode steps at its request
    count, and leaves the verify window and draft cache the eager round does."""
    generator = torch.Generator().manual_seed(2)
    model = ToyDraftModel("cuda")
    model.supports_breakable_draft_extend = False
    graphed, graphed_backend = make_eagle(device="cuda", model=model)
    eager, eager_backend = make_eagle(device="cuda")
    _capture(graphed, graphed_backend, [16, 32], lambda _bucket: [1])
    assert not graphed._first_step_graphs
    cases = [
        ([16], [0], True),
        ([9, 7], [0, 100], True),
        ([20, 3, 1], [7, 0, 300], True),
        ([3, 3, 3, 3], [0, 0, 0, 0], True),
        ([70], [0], True),
        ([12, 12], [64, 0], False),
    ]
    with torch.inference_mode():
        for lengths, prefixes, final in cases:
            inputs = _inputs(lengths, generator)
            prepare_extend(
                graphed, graphed_backend, lengths, prefixes, final=final, ids=inputs[0]
            )
            route = graphed._extend_graph_route(
                len(lengths), extend_input(graphed, lengths, hidden=inputs[2])
            )
            assert route == eagle_module._ExtendRoute(None, True), lengths
            window, forwards = _round(
                graphed, graphed_backend, lengths, prefixes, final, inputs
            )
            want, _ = _round(eager, eager_backend, lengths, prefixes, final, inputs)
            # Step 0 alone runs eager.
            assert forwards == 1
            assert torch.equal(window, want), lengths
            assert torch.equal(graphed_backend.cache[1:], eager_backend.cache[1:])


def test_a_replayed_step_0_feeds_eager_decode_steps_past_their_ladder(
    monkeypatch,
) -> None:
    """A step-0 graph at a request count past the decode steps' ladder hands
    its proposals and hidden rows to eager decode steps through the static
    buffers, and the round leaves the verify window and draft cache the eager
    round does."""
    monkeypatch.setattr(eagle_module, "EXTEND_DECODE_MAX_REQUESTS", 2)
    generator = torch.Generator().manual_seed(3)
    graphed, graphed_backend = make_eagle(device="cuda")
    eager, eager_backend = make_eagle(device="cuda")
    _capture(graphed, graphed_backend, [16], lambda _bucket: [1, 2, 3])
    lengths, prefixes = [5, 2, 6], [0, 40, 9]
    inputs = _inputs(lengths, generator)
    with torch.inference_mode():
        prepare_extend(
            graphed, graphed_backend, lengths, prefixes, final=True, ids=inputs[0]
        )
        route = graphed._extend_graph_route(
            len(lengths), extend_input(graphed, lengths, hidden=inputs[2])
        )
        assert route == eagle_module._ExtendRoute((16, 3), False)
        window, forwards = _round(
            graphed, graphed_backend, lengths, prefixes, True, inputs
        )
        want, _ = _round(eager, eager_backend, lengths, prefixes, True, inputs)
    # The decode steps alone run eager.
    assert forwards == STEPS - 1
    assert torch.equal(window, want)
    assert torch.equal(graphed_backend.cache[1:], eager_backend.cache[1:])


def test_a_publication_recorded_into_the_graph_diverges_from_eager(
    monkeypatch,
) -> None:
    """Recorded into the graph, a decode step's publication clears the step's
    layout memo only at capture: the replayed steps reuse the layout the
    previous step built, so the replay no longer matches the eager round.
    The publications go through run_eager for this reason."""
    monkeypatch.setattr(eagle_module, "run_eager", lambda fn: fn())
    generator = torch.Generator().manual_seed(1)
    graphed, graphed_backend = make_eagle(device="cuda")
    eager, eager_backend = make_eagle(device="cuda")
    _capture(graphed, graphed_backend, [2, 16], lambda _bucket: [1, 2])
    lengths, prefixes = [1, 1], [20, 30]
    inputs = _inputs(lengths, generator)
    with torch.inference_mode():
        window, forwards = _round(
            graphed, graphed_backend, lengths, prefixes, True, inputs
        )
        want, _ = _round(eager, eager_backend, lengths, prefixes, True, inputs)
    assert forwards == 0
    assert not (
        torch.equal(window, want)
        and torch.equal(graphed_backend.cache[1:], eager_backend.cache[1:])
    )


def test_a_replay_waits_for_a_host_load_into_the_draft_cache() -> None:
    """The draft prologue writes K/V inside a captured segment, which holds
    no per-layer Host-load wait, so a step-0 replay waits for the draft
    pool's whole load first. A load still copying into the round's pages
    (delayed on its stream) must land before the replay's writes, not over
    them."""
    sentinel = 1.0e30
    lengths, prefixes = [16], [40]
    pages = slice(1 + prefixes[0], 1 + prefixes[0] + lengths[0])
    generator = torch.Generator().manual_seed(2)
    drafter, backend = make_eagle(device="cuda")
    _capture(drafter, backend, [16], lambda _bucket: [1])
    load_stream = torch.cuda.Stream()

    def round_after_a_slow_load(tracker):
        drafter.token_to_kv_pool.layerwise_load_tracker = tracker
        torch.cuda.synchronize()
        # The load: long-running, then it writes the round's pages and
        # records its last layer's completion.
        with torch.cuda.stream(load_stream):
            torch.cuda._sleep(1_000_000_000)
            backend.cache[pages].fill_(sentinel)
            if tracker is not None:
                tracker.event_sets[0].last_layer_done_event.record(load_stream)
        with torch.inference_mode():
            _, forwards = _round(
                drafter, backend, lengths, prefixes, True, _inputs(lengths, generator)
            )
        torch.cuda.synchronize()
        assert forwards == 0
        return (backend.cache[pages] == sentinel).all(dim=1)

    tracker = LayerwiseLoadTracker(1)
    tracker.set_consumers(0)
    assert not round_after_a_slow_load(tracker).any()
    # ROCm streams can share a hardware queue, which orders the replay after the load.
    if torch.version.hip is not None:
        return
    # Without a tracker nothing orders the replay after the load, which then
    # lands over the replay's writes: the case the wait exists for.
    assert round_after_a_slow_load(None).all()


# ----------------------------------------------------------------------
# A real Qwen4-Exp MTP checkpoint (opt-in).
# ----------------------------------------------------------------------

_MODEL = os.environ.get("TOKENSPEED_DRAFTER_PREFILL_GRAPH_MODEL")
_EXTRA_ARGS = shlex.split(os.environ.get("TOKENSPEED_DRAFTER_PREFILL_GRAPH_ARGS", ""))
# Chunks no wider than the largest bucket, so a mid chunk replays step 0.
_CHUNK = 1024
_SAMPLING = {"max_new_tokens": 12, "temperature": 0}
_CAPTURED = re.compile(
    r"drafter prefill graph: captured step 0 at \(tokens, requests\) "
    r"(\[.*?\]), decode steps at requests (\[.*?\])"
)


def _engine(*, drafter_graph: bool):
    # Imported here: the driver needs the engine's frontend extension.
    from tokenspeed.runtime.entrypoints.wire_driver import WireDriver

    argv = [
        "--model",
        _MODEL,
        "--trust-remote-code",
        "--speculative-algorithm",
        "MTP",
        "--speculative-num-steps",
        "3",
        "--chunked-prefill-size",
        str(_CHUNK),
        "--prefill-graph-max-tokens",
        str(_CHUNK),
        "--prefill-graph-capture-batch-sizes",
        "1",
        "2",
        "3",
        "--max-num-seqs",
        "8",
        "--max-model-len",
        "16384",
        "--gpu-memory-utilization",
        "0.8",
        "--disable-kvstore",
        (
            "--enable-drafter-prefill-graph"
            if drafter_graph
            else "--no-enable-drafter-prefill-graph"
        ),
        *_EXTRA_ARGS,
    ]
    return WireDriver(
        argv, num_engines=1, routing="round_robin", startup_timeout_s=3600
    )


def _workload() -> list[list[list[int]]]:
    """Batches of prompts, each submitted at once (one prefill forward when
    it fits a chunk): one to three requests per forward (each captured),
    four (no graph), token counts across the bucket ladder, a cached prefix,
    and a prompt longer than a chunk (mid chunks replay step 0 alone)."""
    rng = random.Random(2026)

    def prompt(n: int) -> list[int]:
        return [rng.randrange(1000, 20000) for _ in range(n)]

    shared = prompt(700)
    return [
        [prompt(200)],
        [prompt(1)],
        [prompt(16)],
        [prompt(300), prompt(150), prompt(77)],
        [prompt(500), prompt(500)],
        [shared],
        [shared + prompt(90)],
        [shared + prompt(33), prompt(40)],
        [prompt(3000)],
        [prompt(60), prompt(61), prompt(62), prompt(63)],
    ]


def _serve(engine) -> list[list[int]]:
    from tokenspeed.runtime.entrypoints.wire_driver import generate_request

    outputs = []
    for batch in _workload():
        for completion in engine.generate(
            [generate_request(ids, _SAMPLING) for ids in batch]
        ):
            assert completion.finish_reason in ("length", "stop"), (
                completion.finish_reason,
                completion.finish_details,
            )
            outputs.append(completion.output_ids)
    return outputs


@pytest.mark.skipif(_MODEL is None, reason="set TOKENSPEED_DRAFTER_PREFILL_GRAPH_MODEL")
def test_a_real_mtp_model_serves_the_same_tokens_with_the_switch(capfd) -> None:
    """The engine captures drafter graphs, replays them, and serves the
    tokens it serves without the switch: the target decides every token at
    temperature 0, so only a near-tie flipped by different verify rounds
    could change one."""
    with _engine(drafter_graph=True) as engine:
        graphed = _serve(engine)
    log = "".join(capfd.readouterr())
    print("\n".join(line for line in log.splitlines() if "drafter prefill" in line))
    captures = _CAPTURED.findall(log)
    assert captures, "no drafter prefill graphs were captured"
    assert all(first != "[]" and decode != "[]" for first, decode in captures)
    assert "drafter prefill graph ENGAGED" in log

    with _engine(drafter_graph=False) as engine:
        eager = _serve(engine)
    assert graphed == eager
