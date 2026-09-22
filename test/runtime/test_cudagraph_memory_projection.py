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

"""The CUDA-graph reserve: the observer, the projection, and the boot step."""

from __future__ import annotations

import contextlib
import pathlib
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.execution import cudagraph_memory  # noqa: E402
from tokenspeed.runtime.execution import memory_delta  # noqa: E402
from tokenspeed.runtime.execution import model_executor  # noqa: E402
from tokenspeed.runtime.execution.cudagraph_memory import (  # noqa: E402
    PROBE_ENTRIES_PER_LADDER,
    _entry_counts,
    estimate_cudagraph_memory,
    probe_cudagraph_memory,
    reserve_and_rebind,
)
from tokenspeed.runtime.execution.memory_delta import (  # noqa: E402
    NULL_MEMORY_DELTA_OBSERVER,
    DriverMemoryDeltaObserver,
)
from tokenspeed.runtime.execution.model_executor import ModelExecutor  # noqa: E402
from tokenspeed.runtime.layers.attention import utils as attn_utils  # noqa: E402

MIB = 1 << 20
GIB = 1 << 30


def test_the_observer_brackets_each_window_on_the_device_it_was_given() -> None:
    order: list[str] = []
    readings = iter((1000, 900, 900, 850))

    def _on_gpu_2(name, result=None):
        def call(gpu_id=2):
            assert gpu_id == 2
            order.append(name)
            return result() if result else None

        return call

    device = SimpleNamespace(
        synchronize=_on_gpu_2("sync"),
        empty_cache=lambda: order.append("empty"),
        mem_get_info=_on_gpu_2("read", lambda: (next(readings), 2000)),
    )
    observer = DriverMemoryDeltaObserver(device, 2)
    with mock.patch.object(memory_delta.gc, "collect", lambda: order.append("gc")):
        for _ in range(2):
            with observer.measure("phase"):
                pass

    assert observer.samples["phase"] == [100, 50]
    # Collecting after empty_cache biases every delta down.
    assert order == ["sync", "gc", "empty", "read"] * 4

    with NULL_MEMORY_DELTA_OBSERVER.measure("decode"):
        pass
    assert NULL_MEMORY_DELTA_OBSERVER.samples == {}


@pytest.mark.parametrize(
    "samples, entries, expected",
    [
        # (measured, rate, unsampled): disjoint ladders add up.
        (
            {"prefill": (100, 7, 9, 8), "decode": (30, 5, 4)},
            {"prefill": 5, "decode": 3},
            {"prefill": (124, 8, 8), "decode": (39, 5, 0)},
        ),
        # A growth step is spread over the window, not dropped.
        (
            {"prefill": (2530, 40, 650, 40)},
            {"prefill": 48},
            {"prefill": (3260, 244, 244 * 44)},
        ),
        # Memory handed back is neither a credit nor a discount on the divisor.
        ({"prefill": (100, -48, -50, -46)}, {"prefill": 40}, {"prefill": (100, 0, 0)}),
        (
            {"prefill": (100, 38, 38, -38, 38)},
            {"prefill": 40},
            {"prefill": (214, 29, 29 * 35)},
        ),
        ({"prefill": (-100, -5, -6, -7)}, {"prefill": 4}, {"prefill": (0, 0, 0)}),
        # The pool-creating capture can read negative; it is floored, not credited.
        ({"decode": (-48, 26, 30, 26)}, {"decode": 10}, {"decode": (82, 28, 28 * 6)}),
        # A variant's opening capture stays in its own ladder.
        (
            {"decode:a": (100, 10, 10, 10), "decode:b": (500, 10, 10, 10)},
            {"decode:a": 8, "decode:b": 8},
            {"decode:a": (130, 10, 40), "decode:b": (530, 10, 40)},
        ),
        # Two samples are enough for a rate; the tail scales with the ladder.
        ({"prefill": (100, 7)}, {"prefill": 40}, {"prefill": (107, 7, 7 * 38)}),
        ({"decode": (1, 1, 1, 1)}, {"decode": 128}, {"decode": (4, 1, 124)}),
        ({"decode": (0,) * 0}, {"decode": 0}, {"decode": (0, 0, 0)}),
    ],
)
def test_the_projection(samples, entries, expected) -> None:
    estimate = estimate_cudagraph_memory(samples, entries)
    got = {name: tuple(vars(estimate.series[name]).values()) for name in expected}
    assert got == expected
    assert estimate.measured_total == sum(m for m, _r, _u in expected.values())
    assert estimate.unsampled_total == sum(u for _m, _r, u in expected.values())


@pytest.mark.parametrize(
    "samples, entries, match",
    [
        ({"decode": (5,)}, {"decode": 0}, "no entries"),
        ({"prefill": (100,)}, {"prefill": 40}, "expected between 2"),
        ({"prefill": (100, 7, 9)}, {"prefill": 2}, "expected between 2"),
        ({"ghost": (5, 1), "decode": (9, 1)}, {"decode": 2}, "unknown ladders"),
    ],
)
def test_the_projection_refuses_samples_the_capture_cannot_produce(
    samples, entries, match
) -> None:
    with pytest.raises(ValueError, match=match):
        estimate_cudagraph_memory(samples, entries)


def test_the_probe_width_is_measured_not_arbitrary() -> None:
    assert PROBE_ENTRIES_PER_LADDER == 5


def _probe(samples, entries, *, world_size=1, gpu_id=0, hungriest=None):
    """Run the probe with a fabricated observer; returns (reserve, seen)."""
    seen = {"gpu_ids": []}

    class Observer:
        def __init__(self, _device_module, gpu):
            seen["gpu_ids"].append(gpu)
            self.samples = {name: list(v) for name, v in samples.items()}

        @contextlib.contextmanager
        def measure(self, series):
            self.samples.setdefault(series, []).append(0)
            yield

    executor = SimpleNamespace(
        capture_graphs=lambda entries, observer: seen.update(entries=entries),
        drafter=None,
        captures_drafter_prefill_graph=False,
        forward_step=SimpleNamespace(
            capture_entries={k: v for k, v in entries.items() if "decode" in k}
        ),
        prefill_graph=SimpleNamespace(
            capture_entries={k: v for k, v in entries.items() if "decode" not in k}
        ),
    )
    server_args = SimpleNamespace(
        device="cuda",
        mapping=SimpleNamespace(world_size=world_size, world_group="world"),
    )
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(cudagraph_memory, "DriverMemoryDeltaObserver", Observer)
        )
        if hungriest is not None:
            stack.enter_context(
                mock.patch.object(cudagraph_memory, "_hungriest_rank", hungriest)
            )
        return probe_cudagraph_memory(executor, server_args, gpu_id), seen


def test_the_probe_samples_its_own_device_and_reserves_every_ladder() -> None:
    samples = {"prefill": [700, 10, 10, 10], "decode:default": [300, 6, 6, 6]}
    entries = {"prefill": 8, "decode:default": 8}

    reserve, seen = _probe(samples, entries, gpu_id=3)

    assert seen["gpu_ids"] == [3, 3]
    assert seen["entries"] == PROBE_ENTRIES_PER_LADDER
    assert reserve == (730 + 10 * 4) + (318 + 6 * 4)


def test_the_probe_reserves_what_the_reduction_returned() -> None:
    reserve, _ = _probe(
        {"decode:default": [300, 6, 6, 6]},
        {"decode:default": 8},
        world_size=8,
        hungriest=lambda _args, total: total + 777,
    )
    assert reserve == 318 + 6 * 4 + 777


def test_a_ladder_the_probe_could_not_price_warns_the_operator() -> None:
    cases = [
        ([1 << 24, 0, 0, 0, 0], "from 0 non-zero marginals"),
        ([1 << 24, MIB, 0, 0, 0], "from 1 non-zero marginals"),
        ([1 << 24, MIB, 0, MIB, 0], None),
        ([1 << 24, MIB, MIB, MIB, 0], None),
    ]
    for samples, warning in cases:
        with mock.patch.object(cudagraph_memory.logger, "warning") as warn:
            reserve, _ = _probe({"decode:default": samples}, {"decode:default": 40})
        text = " ".join(str(call.args[0]) for call in warn.call_args_list)
        if warning is None:
            assert text == "", samples
            continue
        assert warning in text and "35 unsampled" in text
        assert "--disable-cudagraph-memory-reserve" in text
        if samples[1] == 0:
            assert reserve == 1 << 24


def test_the_probe_never_reserves_less_than_it_spent() -> None:
    # Free MiB per read; the whole-probe bracket sees 700, the two windows 150.
    script = [10_000, 9_950, 9_850, 9_850, 9_800, 9_300]
    reads = iter(range(100))

    device = SimpleNamespace(
        synchronize=lambda _gpu: None,
        empty_cache=lambda: None,
        mem_get_info=lambda _gpu: (script[min(next(reads), 5)] * MIB, 0),
    )

    def capture(*, entries, observer):
        for _ in range(2):
            with observer.measure("decode:default"):
                pass

    executor = SimpleNamespace(
        device="cuda",
        drafter=None,
        captures_drafter_prefill_graph=False,
        prefill_graph=SimpleNamespace(capture_entries={}),
        forward_step=SimpleNamespace(capture_entries={"decode:default": 2}),
        capture_graphs=capture,
    )
    server_args = SimpleNamespace(
        device="cuda", mapping=SimpleNamespace(world_size=1, world_group=None)
    )
    with mock.patch.object(torch, "get_device_module", lambda _d: device):
        assert probe_cudagraph_memory(executor, server_args, 0) == 700 * MIB


def test_the_hungriest_rank_is_the_float64_max_and_one_rank_skips_it(
    monkeypatch,
) -> None:
    calls = []

    def all_reduce(tensor, *, op, group):
        calls.append((tensor.dtype, op))
        tensor.fill_(float((1 << 53) + 2))

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(
        "tokenspeed.runtime.distributed.process_group_manager."
        "process_group_manager.get_process_group",
        lambda *_a: "gloo",
    )
    one = SimpleNamespace(mapping=SimpleNamespace(world_group="w", world_size=1))
    many = SimpleNamespace(mapping=SimpleNamespace(world_group="w", world_size=8))

    assert cudagraph_memory._hungriest_rank(one, 1024) == 1024
    assert calls == []
    assert cudagraph_memory._hungriest_rank(many, 1024) == (1 << 53) + 2
    assert calls == [(torch.float64, torch.distributed.ReduceOp.MAX)]


def test_the_drafters_own_pool_is_declared_captured_and_labelled() -> None:
    seen = []
    drafter = SimpleNamespace(
        captures_prefill_graph=True,
        capture_prefill_graph=lambda _stream, observer: seen.append(observer),
    )
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.device = "cuda"
    executor.drafter = drafter
    executor.forward_step = SimpleNamespace(
        disable=True, capture_entries={}, stream="stream"
    )
    executor.prefill_graph = SimpleNamespace(disable=True, capture_entries={})
    assert _entry_counts(executor) == {}

    executor.prefill_graph = SimpleNamespace(
        disable=False, capture_entries={"prefill": 4}, capture=lambda *a, **k: None
    )
    assert _entry_counts(executor) == {"prefill": 4, "prefill:drafter": 1}

    observer = mock.Mock()
    with mock.patch.object(model_executor, "workspace_pool", lambda _d: mock.Mock()):
        ModelExecutor.capture_graphs(executor, entries=None, observer=observer)
    observer.measure.assert_called_once_with("prefill:drafter")
    assert seen == [observer.measure.return_value]


def test_the_boot_step_measures_releases_rebuilds_then_publishes() -> None:
    order = []
    executor = SimpleNamespace(
        attn_backend="target backend",
        draft_attn_backend="draft backend",
        release_graphs=lambda: order.append(("release",)),
        set_cache_pool=lambda target, draft: order.append(("adopt", target, draft)),
    )
    built = SimpleNamespace(token_to_kv_pool="pool", draft_token_to_kv_pool="draft")

    def build_components(**kwargs):
        order.append(("build", kwargs))
        return built

    with mock.patch.object(
        cudagraph_memory,
        "probe_cudagraph_memory",
        lambda *args: order.append(("probe", *args[1:])) or 4096,
    ):
        rebuilt = reserve_and_rebind(
            executor, build_components, "args", 3, profiled_cache_bytes=9000
        )
    assert rebuilt is built

    assert order == [
        ("probe", "args", 3),
        ("release",),
        (
            "build",
            dict(
                graph_reserve_bytes=4096,
                probe_batch_rows=None,
                profiled_cache_bytes=9000,
                reuse_target_backend="target backend",
                reuse_draft_backend="draft backend",
            ),
        ),
        ("adopt", "pool", "draft"),
    ]


@pytest.mark.parametrize(
    "profiled, reserve, expected",
    [
        (8 * GIB, 0, 8 * GIB),
        (8 * GIB, 1 * GIB, 7 * GIB),
        # Without a reserve the profile's own negative budget passes through.
        (-9 * GIB, 0, -9 * GIB),
        # Exactly zero left raises, and names the escape hatch.
        (1 * GIB, 1 * GIB, ValueError),
        (1 * GIB, 4 * GIB, ValueError),
    ],
)
def test_the_reserve_comes_out_of_the_profiled_budget(profiled, reserve, expected):
    if expected is ValueError:
        with pytest.raises(ValueError, match="disable-cudagraph-memory-reserve"):
            attn_utils.reserve_cache_budget(profiled, reserve)
        return
    assert attn_utils.reserve_cache_budget(profiled, reserve) == expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
