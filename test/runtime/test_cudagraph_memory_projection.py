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
import math
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
    CapturedLadder,
    _ladders,
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


def _top(count, sampled):
    """A ladder of ``count`` widths falling by one, sampled at its top."""
    return CapturedLadder(list(range(count, 0, -1)), list(range(sampled)))


def _rate(positive_sum, marginals):
    """The window's rate: its positive bytes plus a granule over its marginals."""
    return -(-((positive_sum + 2) * MIB) // marginals)


@pytest.mark.parametrize(
    "samples, ladders, expected",
    [
        # (measured, unsampled) in MiB: disjoint ladders add up.
        (
            {"prefill": (100, 7, 9, 8), "decode": (30, 5, 4)},
            {"prefill": _top(5, 4), "decode": _top(3, 3)},
            {"prefill": (124, _rate(24, 3)), "decode": (39, 0)},
        ),
        # A lumpy window is priced at its mean, plus one granule.
        (
            {"decode": (282, 0, 0, 0, 4)},
            {"decode": _top(23, 5)},
            {"decode": (286, 18 * _rate(4, 4))},
        ),
        (
            {"decode": (282, 2, 0, 2, 6)},
            {"decode": _top(23, 5)},
            {"decode": (292, 18 * _rate(10, 4))},
        ),
        # Memory handed back is neither a credit nor a discount on the divisor.
        (
            {"decode": (100, -48, -50, -46)},
            {"decode": _top(40, 4)},
            {"decode": (100, 0)},
        ),
        (
            {"decode": (100, 38, 38, -38, 38)},
            {"decode": _top(40, 5)},
            {"decode": (214, 35 * _rate(114, 4))},
        ),
        ({"decode": (-100, -5, -6, -7)}, {"decode": _top(4, 4)}, {"decode": (0, 0)}),
        # The pool-creating capture can read negative; it is floored, not credited.
        (
            {"decode": (-48, 26, 30, 26)},
            {"decode": _top(10, 4)},
            {"decode": (82, 6 * _rate(82, 3))},
        ),
        # A variant's opening capture stays in its own ladder.
        (
            {"decode:a": (100, 10, 10, 10), "decode:b": (500, 10, 10, 10)},
            {"decode:a": _top(8, 4), "decode:b": _top(8, 4)},
            {"decode:a": (130, 4 * _rate(30, 3)), "decode:b": (530, 4 * _rate(30, 3))},
        ),
        # Two samples are enough for a rate; the tail scales with the ladder.
        (
            {"decode": (100, 7)},
            {"decode": _top(40, 2)},
            {"decode": (107, 38 * _rate(7, 1))},
        ),
        (
            {"decode": (1, 1, 1, 1)},
            {"decode": _top(128, 4)},
            {"decode": (4, 124 * _rate(3, 3))},
        ),
        ({"decode": ()}, {"decode": _top(0, 0)}, {"decode": (0, 0)}),
        # Between the window and an anchor entries sit on the line; below it, flat.
        (
            {"prefill": (200, 20, 20, 10, 4)},
            {
                "prefill": CapturedLadder(
                    [512, 448, 384, 320, 256, 192, 128, 64, 32], [0, 1, 2, 3, 6]
                )
            },
            # Window 52/3 at 384, anchor 4 + 2 at 128: the line at 256 and 192, then 6 + 6.
            {"prefill": (254, math.ceil(24 * MIB + 0.75 * (_rate(50, 3) - 6 * MIB)))},
        ),
        # Inline variants share their bucket's width; the dearer reading anchors it.
        (
            {"prefill": (100, 10, 8, 8, 6, 6, 2, 4)},
            {
                "prefill": CapturedLadder(
                    [64, 64, 48, 48, 32, 32, 16, 16, 8, 8, 4, 4],
                    [0, 1, 2, 3, 4, 5, 8, 9],
                )
            },
            # Window 8 at 44.8, anchor 6 at 8: two on the line at 16, two flat at 4.
            {"prefill": (144, math.ceil(MIB * (2 * (6 + 2 * 8 / 36.8) + 12)))},
        ),
        (
            {"prefill": (100, 10, 8, 8, 6, 6, 4, 2)},
            {
                "prefill": CapturedLadder(
                    [64, 64, 48, 48, 32, 32, 16, 16, 8, 8, 4, 4],
                    [0, 1, 2, 3, 4, 5, 8, 9],
                )
            },
            {"prefill": (144, math.ceil(MIB * (2 * (6 + 2 * 8 / 36.8) + 12)))},
        ),
        # A lumpy anchor never prices above the window: every skipped entry at its rate.
        (
            {"prefill": (100, 4, 4, 4, 10)},
            {
                "prefill": CapturedLadder(
                    [100, 90, 80, 70, 60, 50, 40, 30, 20], [0, 1, 2, 3, 6]
                )
            },
            {"prefill": (122, 4 * _rate(12, 3))},
        ),
        # An anchor well above the window is a dearer entry: its reading less three granules.
        (
            {"prefill": (100, 4, 4, 4, 30)},
            {
                "prefill": CapturedLadder(
                    [100, 90, 80, 70, 60, 50, 40, 30, 20], [0, 1, 2, 3, 6]
                )
            },
            # Window 14/3 at 80, anchor 24 at 40: on the line at 60 and 50, then 24 and 24.
            {
                "prefill": (
                    142,
                    math.ceil(
                        2 * 24 * MIB
                        + (24 * MIB + (_rate(12, 3) - 24 * MIB) * 0.5)
                        + (24 * MIB + (_rate(12, 3) - 24 * MIB) * 0.25)
                    ),
                )
            },
        ),
        # An anchor served from slack, or one that handed memory back, prices at one granule.
        (
            {"prefill": (50, 4, 4, 4, 0)},
            {
                "prefill": CapturedLadder(
                    [100, 90, 80, 70, 60, 50, 40, 30, 20], [0, 1, 2, 3, 6]
                )
            },
            # Window 14/3 at 80, anchor 2 at 40: the line at 60 and 50, then 2 and 2.
            {
                "prefill": (
                    62,
                    math.ceil(8 * MIB + 0.75 * (_rate(12, 3) - 2 * MIB)),
                )
            },
        ),
        (
            {"prefill": (50, 4, 4, 4, -6)},
            {
                "prefill": CapturedLadder(
                    [100, 90, 80, 70, 60, 50, 40, 30, 20], [0, 1, 2, 3, 6]
                )
            },
            {
                "prefill": (
                    62,
                    math.ceil(8 * MIB + 0.75 * (_rate(12, 3) - 2 * MIB)),
                )
            },
        ),
    ],
)
def test_the_projection(samples, ladders, expected) -> None:
    samples = {name: tuple(MIB * s for s in series) for name, series in samples.items()}
    estimate = estimate_cudagraph_memory(samples, ladders)
    got = {name: tuple(vars(estimate.series[name]).values()) for name in expected}
    assert got == {name: (MIB * m, u) for name, (m, u) in expected.items()}
    assert estimate.measured_total == sum(MIB * m for m, _u in expected.values())
    assert estimate.unsampled_total == sum(u for _m, u in expected.values())


def test_a_dearer_reading_never_lowers_the_reserve() -> None:
    ladder = CapturedLadder(
        [512, 448, 384, 320, 256, 192, 128, 64, 32], [0, 1, 2, 3, 6]
    )
    # The last two: a granule more in the window pulls the anchor under the lump cap.
    for base in (
        (200, 20, 20, 20, 20),
        (200, 20, 20, 10, 4),
        (200, 0, 2, 0, 4),
        (200, 0, 0, 0, 7),
        (200, 2, 2, 2, 9),
    ):
        samples = tuple(MIB * s for s in base)
        before = estimate_cudagraph_memory({"prefill": samples}, {"prefill": ladder})
        for i in range(len(samples)):
            dearer = samples[:i] + (samples[i] + 2 * MIB,) + samples[i + 1 :]
            after = estimate_cudagraph_memory({"prefill": dearer}, {"prefill": ladder})
            assert after.unsampled_total + after.measured_total >= (
                before.unsampled_total + before.measured_total
            ), (base, i)


@pytest.mark.parametrize(
    "samples, ladders, match",
    [
        ({"decode": (5,)}, {"decode": _top(0, 0)}, "no entries"),
        ({"prefill": (100,)}, {"prefill": _top(40, 1)}, "at least 2"),
        ({"prefill": (100, 7, 9)}, {"prefill": _top(40, 2)}, "expected the first"),
        (
            {"prefill": (100, 7)},
            {"prefill": CapturedLadder([4, 3, 2], [1, 2])},
            "the first",
        ),
        (
            {"prefill": (100, 7)},
            {"prefill": CapturedLadder([4, 3, 2], [2, 0])},
            "not ladder",
        ),
        (
            {"prefill": (100, 7)},
            {"prefill": CapturedLadder([4, 3, 2], [0, 3])},
            "not ladder",
        ),
        (
            {"ghost": (5, 1), "decode": (9, 1)},
            {"decode": _top(2, 2)},
            "unknown ladders",
        ),
    ],
)
def test_the_projection_refuses_samples_the_capture_cannot_produce(
    samples, ladders, match
) -> None:
    with pytest.raises(ValueError, match=match):
        estimate_cudagraph_memory(samples, ladders)


def _probe(samples, ladders, *, world_size=1, gpu_id=0, hungriest=None):
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
            capture_ladders=lambda entries: {
                k: v for k, v in ladders.items() if "decode" in k
            }
        ),
        prefill_graph=SimpleNamespace(
            capture_ladders=lambda entries: {
                k: v for k, v in ladders.items() if "decode" not in k
            }
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
    samples = {
        "prefill": [MIB * s for s in (700, 10, 10, 10)],
        "decode:default": [MIB * s for s in (300, 6, 6, 6)],
    }
    ladders = {"prefill": _top(4, 4), "decode:default": _top(8, 4)}

    reserve, seen = _probe(samples, ladders, gpu_id=3)

    assert seen["gpu_ids"] == [3]
    assert seen["entries"] == PROBE_ENTRIES_PER_LADDER
    assert reserve == MIB * (730 + 318) + 4 * _rate(18, 3)


def test_the_probe_reserves_what_the_reduction_returned() -> None:
    reserve, _ = _probe(
        {"decode:default": [MIB * s for s in (300, 6, 6, 6)]},
        {"decode:default": _top(8, 4)},
        world_size=8,
        hungriest=lambda _args, total: total + 777,
    )
    assert reserve == MIB * 318 + 4 * _rate(18, 3) + 777


def test_a_ladder_the_probe_could_not_price_warns_the_operator() -> None:
    # One positive reading prices the tail at about a granule; only none at all warns.
    cases = [
        ([1 << 24, 0, 0, 0, 0], "priced its 35 unsampled entries at nothing"),
        ([1 << 24, MIB, 0, 0, 0], None),
        ([1 << 24, MIB, 0, MIB, 0], None),
        ([1 << 24, MIB, MIB, MIB, 0], None),
    ]
    for samples, warning in cases:
        with mock.patch.object(cudagraph_memory.logger, "warning") as warn:
            reserve, _ = _probe(
                {"decode:default": samples}, {"decode:default": _top(40, 5)}
            )
        text = " ".join(str(call.args[0]) for call in warn.call_args_list)
        if warning is None:
            assert text == "", samples
            continue
        assert warning in text
        assert "--disable-cudagraph-memory-reserve" in text
        assert reserve == 1 << 24


def test_memory_taken_between_captures_is_not_reserved() -> None:
    # Free MiB per read: each capture takes 50, and 400 go elsewhere between them.
    script = iter([10_000, 9_950, 9_550, 9_500])

    device = SimpleNamespace(
        synchronize=lambda _gpu: None,
        empty_cache=lambda: None,
        mem_get_info=lambda _gpu: (next(script) * MIB, 0),
    )

    def capture(*, entries, observer):
        for _ in range(2):
            with observer.measure("decode:default"):
                pass

    executor = SimpleNamespace(
        device="cuda",
        drafter=None,
        captures_drafter_prefill_graph=False,
        prefill_graph=SimpleNamespace(capture_ladders=lambda entries: {}),
        forward_step=SimpleNamespace(
            capture_ladders=lambda entries: {"decode:default": _top(2, 2)}
        ),
        capture_graphs=capture,
    )
    server_args = SimpleNamespace(
        device="cuda", mapping=SimpleNamespace(world_size=1, world_group=None)
    )
    with mock.patch.object(torch, "get_device_module", lambda _d: device):
        assert probe_cudagraph_memory(executor, server_args, 0) == 100 * MIB


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
        disable=True, capture_ladders=lambda entries: {}, stream="stream"
    )
    executor.prefill_graph = SimpleNamespace(
        disable=True, capture_ladders=lambda entries: {}
    )
    assert _ladders(executor, 5) == {}

    executor.prefill_graph = SimpleNamespace(
        disable=False,
        capture_ladders=lambda entries: {"prefill": _top(4, entries)},
        capture=lambda *a, **k: None,
    )
    assert _ladders(executor, 3) == {
        "prefill": _top(4, 3),
        "prefill:drafter": CapturedLadder((1,), (0,)),
    }

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
