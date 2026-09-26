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

"""What the probe samples from each ladder, what it counts, and what it releases."""

from __future__ import annotations

import ast
import contextlib
import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.execution import cudagraph_memory  # noqa: E402
from tokenspeed.runtime.execution import device  # noqa: E402
from tokenspeed.runtime.execution import forward_step  # noqa: E402
from tokenspeed.runtime.execution import prefill_graph  # noqa: E402
from tokenspeed.runtime.execution.cudagraph_memory import (  # noqa: E402
    CapturedLadder,
    probe_positions,
)
from tokenspeed.runtime.execution.forward_step import ForwardStepRunner  # noqa: E402
from tokenspeed.runtime.execution.memory_delta import (  # noqa: E402
    NULL_MEMORY_DELTA_OBSERVER,
)
from tokenspeed.runtime.execution.prefill_graph import PrefillGraph  # noqa: E402
from tokenspeed.runtime.layers.attention import registry  # noqa: E402

WIDTH = cudagraph_memory.PROBE_ENTRIES_PER_LADDER


class _CountingObserver:
    """Records 0, 1, 2, ... per measured region of a series."""

    def __init__(self) -> None:
        self.samples: dict[str, list[int]] = {}

    @contextlib.contextmanager
    def measure(self, series: str):
        yield
        self.samples.setdefault(series, []).append(len(self.samples.get(series, [])))


def _enter(observer, result):
    with observer:
        return result


def _decode_runner(capture_bs, variants=None):
    """A ForwardStepRunner with only what ``capture`` reads."""
    runner = ForwardStepRunner.__new__(ForwardStepRunner)
    runner.capture_bs = list(capture_bs)
    runner.global_rank = 1
    runner.enable_cudagraph_gc = False
    runner.disable = False
    runner.max_tokens_per_req = 1
    runner.graphs = {}
    runner.output_buffers = {}
    runner._placeholder_tables = {}
    runner.sampling_backend = variants and SimpleNamespace(
        cuda_graph_capture_variants=lambda _n: tuple(variants)
    )
    runner._capture_one = lambda bs, *, variant, observer: _enter(
        observer, ((variant, bs), f"out{bs}")
    )
    return runner


def _prefill_graph(buckets, *, decoder_buckets=None, context_len=4096, sizes=None):
    """A PrefillGraph with only what the capture loops read."""
    graph = PrefillGraph.__new__(PrefillGraph)
    graph.disable = False
    graph.capture_buckets = list(buckets)
    graph.decoder_buckets = list(decoder_buckets or ())
    graph._narrowing = decoder_buckets and SimpleNamespace(
        max_decoder_rows_per_request=4,
        allocate_decoder_state=lambda rows: ("statics", rows),
        decoder_rows=lambda ctx: ctx.rows,
        narrowing_forward=lambda _encoded, _ctx: SimpleNamespace(
            land_into=lambda _statics: None
        ),
    )
    graph.config = SimpleNamespace(
        global_rank=1,
        device="cpu",
        gpu_id=0,
        context_len=context_len,
        max_num_seqs=8,
        data_parallel_size=1,
        prefill_graph_capture_batch_sizes=sizes,
    )
    graph.dp_size = 1
    graph.reset_backend = []
    graph.dummies = []
    graph.attn_backend = SimpleNamespace(
        admits_prefill_graph=lambda *_a: True,
        prepare_prefill_metadata=lambda *_a, **_k: True,
        init_prefill_graph_state=lambda **kwargs: graph.reset_backend.append(kwargs),
    )
    graph.input_buffers = SimpleNamespace(
        input_ids_buf=torch.zeros(max(buckets), dtype=torch.int32)
    )
    graph._embed_tokens = lambda ids: ids
    graph._land_input_embeds = lambda _embeds, _bucket: None
    graph.make_dummy_batch = lambda rows, bs: graph.dummies.append((rows, bs)) or (
        SimpleNamespace(capture_hidden_mode=None, forward_mode=None, rows=rows)
    )
    graph._captures = {}
    graph._encoders = {}
    graph._decoders = {}
    graph._run_encoder = lambda rows: ("encoded", rows)
    graph._capture_bucket = lambda bucket, _wrapper, observer: _enter(observer, bucket)
    graph._capture_encoder = graph._capture_bucket
    graph._capture_decoder = lambda statics, _rearm, _wrapper, observer: _enter(
        observer, statics
    )
    return graph


@pytest.mark.parametrize("variants", [None, ("default", "penalties")])
def test_the_decode_probe_samples_the_probe_positions_of_each_variant(variants) -> None:
    runner = _decode_runner([2**i for i in range(WIDTH + 2)], variants)
    names = [f"decode:{v}" for v in (variants or ("default",))]
    ladder = sorted((2**i for i in range(WIDTH + 2)), reverse=True)
    # Seven entries: the widest three, then position 4 (the third falls among them).
    assert runner.capture_ladders(WIDTH) == {
        name: CapturedLadder(ladder, [0, 1, 2, 4]) for name in names
    }

    observer = _CountingObserver()
    runner.capture(entries=WIDTH, observer=observer)

    for variant in variants or ("default",):
        assert sorted(bs for v, bs in runner.graphs if v == variant) == [4, 16, 32, 64]
    estimate = cudagraph_memory.estimate_cudagraph_memory(
        observer.samples, runner.capture_ladders(WIDTH)
    )
    granule = 2 << 20
    window = -(-(3 + granule) // 2)
    # The anchor's reading plus a granule exceeds the window's rate, so it is capped there.
    for name in names:
        assert observer.samples[name] == [0, 1, 2, 3]
        assert estimate.series[name].unsampled == 3 * window

    serving = _decode_runner([1, 2, 4], variants)
    serving.capture(entries=None, observer=NULL_MEMORY_DELTA_OBSERVER)
    assert len(serving.graphs) == 3 * len(names)
    assert serving.capture_ladders(None) == {
        name: CapturedLadder([4, 2, 1], [0, 1, 2]) for name in names
    }

    serving.disable = True
    assert serving.capture_ladders(None) == {}


def test_the_probe_positions_are_the_widest_few_and_two_down_the_ladder() -> None:
    assert probe_positions(40, 5) == [0, 1, 2, 13, 26]
    assert probe_positions(56, 5) == [0, 1, 2, 18, 37]
    assert probe_positions(7, 5) == [0, 1, 2, 4]
    assert probe_positions(5, 5) == [0, 1, 2, 3, 4]
    assert probe_positions(3, 5) == [0, 1, 2]
    assert probe_positions(40, None) == list(range(40))


def test_the_prefill_probe_samples_buckets_with_their_inline_variants() -> None:
    graph = _prefill_graph([2 ** (i + 2) for i in range(WIDTH + 2)], sizes=[1, 2, 4])
    buckets = sorted(graph.capture_buckets, reverse=True)
    widths = [b for b in buckets for _ in range(4)]
    assert graph.capture_ladders(None) == {
        "prefill": CapturedLadder(widths, list(range(4 * (WIDTH + 2))))
    }
    # Buckets at positions 0, 1, 2 and 4 of seven; each brings its three variants.
    sampled = [
        i for i, b in enumerate(widths) if b in {buckets[j] for j in (0, 1, 2, 4)}
    ]
    assert graph.capture_ladders(WIDTH) == {"prefill": CapturedLadder(widths, sampled)}

    observer = _CountingObserver()
    graph._capture_all_buckets(None, WIDTH, observer)

    assert set(graph._captures) == {
        (b, bs) for b in (buckets[j] for j in (0, 1, 2, 4)) for bs in (None, 1, 2, 4)
    }
    assert len(observer.samples["prefill"]) == 16
    estimate = cudagraph_memory.estimate_cudagraph_memory(
        observer.samples, graph.capture_ladders(WIDTH)
    )
    assert estimate.unsampled_total > 0

    graph._captures.clear()
    graph._capture_all_buckets(None, None, _CountingObserver())
    assert set(graph._captures) == set(graph.capture_plan["prefill"])

    graph.dp_size = 2
    assert graph.capture_ladders(None) == {
        "prefill": CapturedLadder(buckets, list(range(WIDTH + 2)))
    }
    graph.attn_backend.admits_prefill_graph = lambda *_a: False
    graph.dp_size = 1
    assert graph.capture_plan == {
        "prefill": [(b, None) for b in reversed(graph.capture_buckets)]
    }
    graph.disable = True
    assert graph.capture_ladders(None) == {}


def test_a_bucket_captures_only_the_inline_counts_it_admits() -> None:
    # A bucket admits a count only from ceil(bucket / context_len) upwards.
    graph = _prefill_graph([64, 128, 256, 512, 1024], context_len=64, sizes=[1, 2, 4])

    graph._capture_all_buckets(None, None, _CountingObserver())

    assert set(graph._captures) == {
        (1024, None),
        (512, None),
        (256, None),
        (256, 4),
        (128, None),
        (128, 2),
        (128, 4),
        (64, None),
        (64, 1),
        (64, 2),
        (64, 4),
    }
    # Each ordinary capture fabricates the fewest requests its bucket needs.
    assert graph.dummies[:3] == [(1024, 16), (512, 8), (256, 4)]


def test_narrowing_declares_and_samples_its_two_ladders() -> None:
    graph = _prefill_graph([64, 128, 256], decoder_buckets=[8, 16, 32, 64])
    assert graph.capture_ladders(None) == {
        "prefill:encoder": CapturedLadder([256, 128, 64], [0, 1, 2]),
        "prefill:decoder": CapturedLadder([64, 32, 16, 8], [0, 1, 2, 3]),
    }
    assert graph.capture_ladders(3) == {
        "prefill:encoder": CapturedLadder([256, 128, 64], [0, 1, 2]),
        "prefill:decoder": CapturedLadder([64, 32, 16, 8], [0, 1, 2]),
    }

    observer = _CountingObserver()
    graph._capture_decoders(None, 3, observer)

    assert sorted(graph._decoders, reverse=True) == [64, 32, 16]
    assert len(observer.samples["prefill:decoder"]) == 3


def test_releasing_drops_the_graphs_and_pools_but_keeps_the_tables(monkeypatch) -> None:
    graph = _prefill_graph([64, 128], decoder_buckets=[8, 16])
    graph._captures = {64: object()}
    graph._encoders = {64: object()}
    graph._decoders = {8: object()}
    graph._pool = object()

    graph.release_graphs()

    assert (graph._captures, graph._encoders, graph._decoders) == ({}, {}, {})
    assert graph._pool is None

    runner = _decode_runner([1, 2])
    runner._metadata_snapshots = {("default", 1): object()}
    runner._placeholder_tables = {"history": object()}
    runner.capture(entries=None, observer=NULL_MEMORY_DELTA_OBSERVER)
    monkeypatch.setattr(forward_step, "global_graph_memory_pool", object())

    runner.release_graphs()

    assert forward_step.global_graph_memory_pool is None
    assert (runner.graphs, runner.output_buffers, runner._metadata_snapshots) == (
        {},
        {},
        {},
    )
    # The tables name no arena, so they stay charged to the coming profile.
    assert runner._placeholder_tables


def test_every_capture_opens_its_observer_after_the_warmups_and_before_the_pool(
    monkeypatch,
) -> None:
    order = []

    class _Recorder:
        def __init__(self, name, pool=None, _stream=None):
            self.name, self.pool = name, pool or "pool"
            if name == "capture":
                order.append("allocate pool")

        def __enter__(self):
            order.append(f"{self.name} enter")
            return self

        def __exit__(self, *_exc):
            order.append(f"{self.name} exit")

        def replay(self):
            order.append("replay")

    monkeypatch.setattr(
        prefill_graph,
        "BreakableCapture",
        lambda pool, stream: _Recorder("capture", pool),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_a, **_k: None)
    graph = PrefillGraph.__new__(PrefillGraph)
    graph.num_warmup = 2
    graph._pool = None
    graph._run_inner = lambda _bucket: order.append("forward") or (torch.zeros(1), None)

    PrefillGraph._capture_bucket(graph, 8, None, _Recorder("observe"))

    assert order == [
        "forward",
        "forward",
        "allocate pool",
        "observe enter",
        "capture enter",
        "forward",
        "capture exit",
        "observe exit",
        "replay",
    ]

    # The same order at every other capture site.
    root = pathlib.Path(prefill_graph.__file__).parent
    found = {}
    for path in (
        root / "prefill_graph.py",
        root / "forward_step.py",
        root / "drafter/deepseek_v4_dspark.py",
    ):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.With):
                spelled = [ast.unparse(item.context_expr) for item in node.items]
                if any("observer" in text for text in spelled):
                    found[path.name] = found.get(path.name, 0) + 1
                    assert spelled[0] == "observer", (path.name, node.lineno)
    assert found == {
        "prefill_graph.py": 3,
        "forward_step.py": 1,
        "deepseek_v4_dspark.py": 1,
    }


def test_a_rebind_hands_back_views_of_the_rebuilt_pool(monkeypatch) -> None:
    import tokenspeed.runtime.engine.scheduler_utils as scheduler_utils

    monkeypatch.setattr(
        scheduler_utils, "scheduler_cache_geometry_from_pool", lambda p: ("geometry", p)
    )
    monkeypatch.setattr(
        scheduler_utils, "pool_to_cache_groups", lambda p: ("groups", p)
    )
    rebuilt = SimpleNamespace(token_to_kv_pool="real", draft_token_to_kv_pool="draft")
    seen = []
    args = SimpleNamespace(
        attention_backend="hybrid_linear_attn", drafter_attention_backend=None
    )
    monkeypatch.setattr(
        cudagraph_memory,
        "reserve_and_rebind",
        lambda *a, profiled_cache_bytes: seen.append(
            (
                a,
                profiled_cache_bytes,
                args.attention_backend,
                args.drafter_attention_backend,
            )
        )
        or rebuilt,
    )
    probe = SimpleNamespace(
        token_to_kv_pool="probe",
        draft_token_to_kv_pool="probe draft",
        profiled_cache_bytes=7,
    )

    attention, views = device._rebind_under_reserve(
        "executor", "build", args, 3, probe, ("trtllm_mla", "flashinfer")
    )

    assert attention is rebuilt
    # The rebuild resolves from the operator's choice, not the probe's write-back.
    assert seen == [(("executor", "build", args, 3), 7, "trtllm_mla", "flashinfer")]
    assert views == device.PoolViews(
        token_to_kv_pool="real",
        draft_token_to_kv_pool="draft",
        cache_geometry=("geometry", "real"),
        cache_groups=("groups", "real"),
    )


@pytest.mark.parametrize("side", ["target", "draft"])
def test_the_replacement_arena_reuses_the_backend_it_was_given(
    monkeypatch, side
) -> None:
    pool, existing = object(), object()
    monkeypatch.setattr(registry, "create_cache_pool", lambda *a, **k: pool)
    for builder in ("_create_attn_backend", "_create_hybrid_linear_attn_backend"):
        monkeypatch.setattr(registry, builder, lambda *a, **k: pytest.fail("rebuilt"))
    common = dict(
        backend=existing,
        server_args=None,
        cache_spec=SimpleNamespace(layer_types=("full_attention",)),
        full_attn_backend_name=None,
        linear_attention=None,
        is_inkling=False,
    )
    if side == "target":
        built = registry._create_target_components(
            model_config=None, config=None, arena=None, rank=0, **common
        )
    else:
        built = registry._create_draft_components(
            model_config=SimpleNamespace(num_attention_layers=1),
            config=SimpleNamespace(),
            pool=SimpleNamespace(arena=None, rank=0),
            num_target_layers=0,
            is_heterogeneous=False,
            **common,
        )
    assert built == (existing, pool)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
