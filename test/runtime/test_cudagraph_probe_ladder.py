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

"""What the probe actually samples, and what it extrapolates over.

The projection is only as good as the two numbers it is fed: which entries of
each ladder the sample came from, and how many entries the full ladder has.
Both are computed by the graph owners themselves, so they are pinned here
against the real ``capture`` bodies rather than against the arithmetic alone.
"""

from __future__ import annotations

import ast
import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.execution import cudagraph_memory  # noqa: E402
from tokenspeed.runtime.execution.forward_step import ForwardStepRunner  # noqa: E402
from tokenspeed.runtime.execution.memory_delta import (  # noqa: E402
    NULL_MEMORY_DELTA_OBSERVER,
)
from tokenspeed.runtime.execution.prefill_graph import PrefillGraph  # noqa: E402


class _CountingObserver:
    """Records one entry per measured region, with no device involved."""

    def __init__(self) -> None:
        self.samples: dict[str, list[int]] = {}

    def measure(self, series: str):
        from contextlib import contextmanager

        @contextmanager
        def _region():
            yield
            self.samples.setdefault(series, []).append(
                len(self.samples.get(series, []))
            )

        return _region()


def _decode_runner(capture_bs, variants):
    """A ForwardStepRunner with only what ``capture`` reads."""
    runner = ForwardStepRunner.__new__(ForwardStepRunner)
    runner.capture_bs = list(capture_bs)
    # Non-zero rank: no tqdm, no device free-memory query.
    runner.global_rank = 1
    runner.enable_cudagraph_gc = False
    runner.disable = False
    runner.max_tokens_per_req = 1
    runner.graphs = {}
    runner.output_buffers = {}
    runner._placeholder_tables = {}
    runner.sampling_backend = (
        None
        if variants is None
        else SimpleNamespace(cuda_graph_capture_variants=lambda _n: tuple(variants))
    )
    runner._capture_one = lambda bs, variant, observer: (
        observer.__enter__(),
        observer.__exit__(None, None, None),
        ((variant, bs), f"out{bs}"),
    )[-1]
    return runner


def _prefill_graph(buckets):
    """A PrefillGraph with only what ``_capture_all_buckets`` reads."""
    graph = PrefillGraph.__new__(PrefillGraph)
    graph.disable = False
    graph.capture_buckets = list(buckets)
    graph.config = SimpleNamespace(global_rank=1, device="cpu", gpu_id=0)
    graph.input_buffers = SimpleNamespace(
        input_ids_buf=torch.zeros(max(buckets), dtype=torch.int32)
    )
    graph._embed_tokens = lambda ids: ids
    graph._land_input_embeds = lambda _embeds, _bucket: None
    graph.make_dummy_batch = lambda _bucket: SimpleNamespace(capture_hidden_mode=None)
    graph._captures = {}
    graph.captured = []
    graph._capture_bucket = lambda bucket, _wrapper, observer: (
        observer.__enter__(),
        observer.__exit__(None, None, None),
        graph.captured.append(bucket),
    )[-1]
    return graph


def test_the_fabricated_split_never_leaves_a_row_over_the_context() -> None:
    """A floored split hands the last request more tokens than the context.

    ``make_dummy_batch`` fabricates positions 0..context_len-1 per row, so a
    row carrying more than that runs its positions past the rope tables --
    and the same count is the probe arena's floor, so it must round up.
    """
    from tokenspeed.runtime.execution.prefill_graph import dummy_batch_size

    for num_tokens, context_len in ((8200, 4096), (4097, 4096), (12289, 4096)):
        rows = dummy_batch_size(num_tokens, context_len)

        assert rows * context_len >= num_tokens, (num_tokens, context_len)
        assert (rows - 1) * context_len < num_tokens, (num_tokens, context_len)


def test_releasing_the_prefill_buckets_drops_the_private_pool() -> None:
    """The next capture must allocate a fresh pool, not the released buckets'.

    The buckets recorded the probe arena's buffers and the embeds buffer they
    were captured from; a pool handle left behind hands the real capture
    blocks those dropped graphs still name.
    """
    graph = _prefill_graph([64, 128])
    graph._captures = {64: object(), 128: object()}
    graph._outputs = {64: object()}
    graph._pool = object()
    graph._input_embeds_buf = object()
    graph._captured_hidden_mode = object()

    graph.release_graphs()

    assert graph._captures == {}
    assert graph._outputs == {}
    assert graph._pool is None
    assert graph._input_embeds_buf is None
    assert graph._captured_hidden_mode is None


def test_the_decode_probe_samples_the_largest_entries() -> None:
    """A projection from the cheapest graphs would under-reserve the ladder.

    The whole ladder is extrapolated from the sampled marginals, so the sample
    has to come off the expensive end: the largest batch sizes are the ones
    whose pool growth the reserve is meant to cover.
    """
    observer = _CountingObserver()
    runner = _decode_runner([1, 2, 4, 8, 16], variants=None)

    runner.capture(entries=2, observer=observer)

    assert sorted(bs for _variant, bs in runner.graphs) == [8, 16]
    assert len(observer.samples["decode:default"]) == 2


def test_the_prefill_probe_samples_the_largest_buckets() -> None:
    """Same contract on the extend side: the widest buckets, not the cheapest."""
    observer = _CountingObserver()
    graph = _prefill_graph([64, 128, 256, 512])

    graph._capture_all_buckets(None, 2, observer)

    assert graph.captured == [512, 256]
    assert len(observer.samples["prefill"]) == 2


def test_a_full_capture_records_the_whole_decode_ladder() -> None:
    """``entries=None`` must stay the serving capture, untouched by the probe."""
    runner = _decode_runner([1, 2, 4], variants=None)

    runner.capture(entries=None, observer=NULL_MEMORY_DELTA_OBSERVER)

    assert sorted(bs for _variant, bs in runner.graphs) == [1, 2, 4]


def test_decode_entry_count_counts_every_sampler_variant() -> None:
    """The ladder the projection extrapolates over is bs x sampler variant.

    ``capture`` records one graph per (variant, bs); counting only the batch
    sizes would extrapolate half the entries and halve the reserve.
    """
    observer = _CountingObserver()
    runner = _decode_runner([1, 2, 4, 8], variants=("default", "penalties"))

    assert runner.capture_entries == {"decode:default": 4, "decode:penalties": 4}

    runner.capture(entries=2, observer=observer)

    assert len(runner.graphs) == 4
    # One series per variant: a variant's opening capture is its own first.
    assert len(observer.samples["decode:default"]) == 2
    assert len(observer.samples["decode:penalties"]) == 2


def test_a_disabled_family_declares_no_entries() -> None:
    """A family that captures nothing must project nothing, not a first entry."""
    runner = _decode_runner([1, 2, 4], variants=None)
    runner.disable = True
    graph = _prefill_graph([64, 128])
    graph.disable = True

    assert runner.capture_entries == {}
    assert graph.capture_entries == {}


@pytest.mark.parametrize("variants", [None, ("default", "penalties")])
def test_the_sample_and_the_entry_count_stay_consistent_for_the_probe(
    variants,
) -> None:
    """The probe's own sample must satisfy the projection's own contract.

    ``_estimate_series`` rejects a sample larger than its ladder, and needs
    two to extrapolate from, so the number of captures the probe takes and
    the number of entries each ladder declares have to be derived from the
    same count -- including the single-variant ladder, where the slice is
    the whole of the sample.
    """
    observer = _CountingObserver()
    runner = _decode_runner([1, 2, 4, 8, 16], variants=variants)

    runner.capture(entries=cudagraph_memory.PROBE_ENTRIES_PER_LADDER, observer=observer)

    estimate = cudagraph_memory.estimate_cudagraph_memory(
        observer.samples, runner.capture_entries
    )

    assert set(observer.samples) == set(runner.capture_entries)
    assert estimate.total >= 0


def test_the_reserve_is_the_hungriest_ranks(monkeypatch) -> None:
    """Every rank has to size its KV cache against the same reserve.

    The ranks measure different deltas; sizing each rank's cache against its
    own would give the ranks different block counts and a scheduler that
    plans one geometry while a peer holds another. The max is the only safe
    reduction -- a min would hand the hungriest rank a budget it cannot pay.
    """
    recorded = {}

    def _all_reduce(tensor, op, group):
        recorded["op"] = op
        recorded["group"] = group
        tensor.fill_(4096.0)

    monkeypatch.setattr(torch.distributed, "all_reduce", _all_reduce)
    monkeypatch.setattr(
        cudagraph_memory,
        "_hungriest_rank",
        cudagraph_memory._hungriest_rank,
    )
    import tokenspeed.runtime.distributed.process_group_manager as pgm

    monkeypatch.setattr(
        pgm.process_group_manager,
        "get_process_group",
        lambda backend, group: f"{backend}:{group}",
        raising=False,
    )
    server_args = SimpleNamespace(
        mapping=SimpleNamespace(world_group="world", world_size=8)
    )

    reduced = cudagraph_memory._hungriest_rank(server_args, 1024)

    assert reduced == 4096
    assert recorded["op"] is torch.distributed.ReduceOp.MAX
    assert recorded["group"] == "gloo:world"


def test_a_single_rank_needs_no_reduction(monkeypatch) -> None:
    def _explode(*_args, **_kwargs):
        raise AssertionError("a single rank must not join a collective")

    monkeypatch.setattr(torch.distributed, "all_reduce", _explode)
    server_args = SimpleNamespace(
        mapping=SimpleNamespace(world_group="world", world_size=1)
    )

    assert cudagraph_memory._hungriest_rank(server_args, 1024) == 1024


def test_releasing_the_decode_graphs_drops_the_shared_mempool_handle(
    monkeypatch,
) -> None:
    """The next capture must start a fresh pool, not reuse released blocks.

    ``global_graph_memory_pool`` is module state: left set, the capture that
    follows a rebind hands CUDA a pool handle whose blocks the dropped graphs
    still name.
    """
    import tokenspeed.runtime.execution.forward_step as forward_step

    runner = _decode_runner([1, 2], variants=None)
    runner._metadata_snapshots = {("default", 1): object()}
    runner._placeholder_tables = {"history": object()}
    runner.capture(entries=None, observer=NULL_MEMORY_DELTA_OBSERVER)
    monkeypatch.setattr(forward_step, "global_graph_memory_pool", object())

    runner.release_graphs()

    assert forward_step.global_graph_memory_pool is None
    assert runner.graphs == {}
    assert runner.output_buffers == {}
    assert runner._metadata_snapshots == {}
    # Charged to the profile that sizes the replacement arena if held.
    assert runner._placeholder_tables == {}


@pytest.mark.parametrize("side", ["target", "draft"])
def test_the_replacement_arena_reuses_the_backend_it_was_given(
    monkeypatch, side
) -> None:
    """The probe measured with these backends; the real pool must land on them.

    Building a second backend would leave the executor rebinding pools onto
    the tree it still owns while the freshly built one -- and the graph state
    it just allocated -- leaks.
    """
    import tokenspeed.runtime.layers.attention.registry as registry

    pool = object()
    monkeypatch.setattr(registry, "create_cache_pool", lambda *a, **k: pool)
    monkeypatch.setattr(
        registry,
        "_create_attn_backend",
        lambda *a, **k: pytest.fail("a reused backend must not be rebuilt"),
    )
    monkeypatch.setattr(
        registry,
        "_create_hybrid_linear_attn_backend",
        lambda *a, **k: pytest.fail("a reused backend must not be rebuilt"),
    )
    existing = object()

    if side == "target":
        returned, returned_pool = registry._create_target_components(
            backend=existing,
            server_args=None,
            model_config=None,
            config=None,
            cache_spec=SimpleNamespace(layer_types=("full_attention",)),
            arena=None,
            rank=0,
            full_attn_backend_name=None,
            is_hybrid_linear=False,
            is_kda=False,
            is_inkling=False,
        )
    else:
        returned, returned_pool = registry._create_draft_components(
            backend=existing,
            server_args=None,
            model_config=SimpleNamespace(num_attention_layers=1),
            config=SimpleNamespace(),
            pool=SimpleNamespace(arena=None, rank=0),
            cache_spec=SimpleNamespace(layer_types=("full_attention",)),
            num_target_layers=0,
            full_attn_backend_name=None,
            is_heterogeneous=False,
            is_hybrid_linear=False,
            is_kda=False,
            is_inkling=False,
        )

    assert returned is existing
    assert returned_pool is pool


def _probe_predicate() -> ast.Return:
    """The body of the one function that decides whether a boot probes."""
    path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "python/tokenspeed/runtime/execution/device.py"
    )
    predicate = next(
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef)
        and node.name == "_can_probe_cudagraph_memory"
    )
    return next(
        node.value for node in ast.walk(predicate) if isinstance(node, ast.Return)
    )


def _build_device_side() -> ast.FunctionDef:
    path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "python/tokenspeed/runtime/execution/device.py"
    )
    return next(
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "build_device_side"
    )


def _probing_branch() -> ast.If:
    return next(
        node
        for node in ast.walk(_build_device_side())
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "probing"
    )


@pytest.mark.parametrize(
    "call", ["scheduler_cache_geometry_from_pool", "pool_to_cache_groups"]
)
def test_the_rebind_republishes_the_scheduler_geometry(call: str) -> None:
    """The scheduler must plan against the real arena, not the probe's.

    ``cache_geometry`` and ``cache_groups`` are read off the pool built for
    the probe. Leaving them there ships the probe arena's parent count to the
    scheduler, which would then admit against a handful of blocks while the
    pool holds thousands.
    """
    names = [
        node.func.id
        for node in ast.walk(_probing_branch())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]

    assert call in names


def test_the_boots_that_cannot_use_a_probe_skip_it() -> None:
    """Two flags and one family fact decide it, and the predicate names them.

    ``enforce_eager`` disables both graph owners (``ForwardStepRunner.disable``
    and ``PrefillGraph.disable`` each start with it), so the probe would bind a
    second arena, tune, and project zero. A family whose speculative verify
    scratch is the bound pool itself is asked, through the factory, rather
    than guessed from a flag: every other speculative boot probes, and so does
    a memory-saver boot (measured: same cost as without it), so
    ``--gpu-memory-utilization`` means one thing.
    """
    probing = _probe_predicate()
    flags = {
        operand.attr
        for operand in ast.walk(probing)
        if isinstance(operand, ast.Attribute)
        and isinstance(operand.value, ast.Name)
        and operand.value.id == "server_args"
    }
    calls = {
        getattr(node.func, "id", None)
        for node in ast.walk(probing)
        if isinstance(node, ast.Call)
    }

    assert flags == {"disable_cudagraph_memory_reserve", "enforce_eager"}
    assert "cudagraph_probe_supported" in calls
    assert isinstance(probing.op, ast.And)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
