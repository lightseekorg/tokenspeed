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

"""Full garbage collections run between rounds, in the same round on every rank.

``freeze_startup_heap`` takes the objects alive at the end of startup out of
every later collection and stops CPython from starting full collections by
itself; ``GcHooks`` runs one every N forwarded rounds. The tests run in this
process and restore the collector afterwards. They check both halves, and that
reference cycles serving drops after they reached the oldest generation are
still freed: by the hooks' collection and by a manual ``gc.collect()``.
"""

from __future__ import annotations

import gc
import os
import sys
import weakref

import pytest

# CPU-only tests scheduled in runtime-1gpu with the other event-loop tests.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.engine import gc_hooks  # noqa: E402
from tokenspeed.runtime.engine.gc_hooks import (  # noqa: E402
    GcHooks,
    freeze_startup_heap,
)


class _Cycle:
    """A request-like object in a reference cycle: only a collection frees it."""

    def __init__(self) -> None:
        self.tokens = list(range(64))
        self.me = self


@pytest.fixture
def serving_collector():
    """The collector as ``freeze_startup_heap`` leaves it; restored afterwards."""
    thresholds = gc.get_threshold()
    freeze_startup_heap()
    yield
    gc.unfreeze()
    gc.set_threshold(*thresholds)
    gc.collect()


def _promote_to_oldest_generation() -> None:
    # A young collection moves every survivor of generations 0 and 1 into
    # generation 2, as a serving round's long-lived objects end up.
    gc.collect(1)


def test_freeze_takes_startup_objects_out_and_raises_only_gen2() -> None:
    startup = [_Cycle() for _ in range(100)]
    thresholds = gc.get_threshold()
    try:
        freeze_startup_heap()

        threshold0, threshold1, threshold2 = gc.get_threshold()
        assert (threshold0, threshold1) == thresholds[:2]
        assert threshold2 == gc_hooks._SERVING_GEN2_THRESHOLD > thresholds[2]
        assert gc.get_freeze_count() >= len(startup)
        # No generation holds them any more, so no collection walks them.
        tracked = {id(obj) for obj in gc.get_objects()}
        assert not any(id(obj) in tracked for obj in startup)
    finally:
        gc.unfreeze()
        gc.set_threshold(*thresholds)
        gc.collect()


def test_hooks_collect_every_interval_of_forwarded_rounds(serving_collector) -> None:
    hooks = GcHooks(full_collection_interval=3)
    full_collections: list[int] = []
    round_index = 0

    def record(phase: str, info: dict) -> None:
        if phase == "stop" and info["generation"] == 2:
            full_collections.append(round_index)

    gc.callbacks.append(record)
    try:
        forwarded_by_round = [True, False, True, True, False, True, True, True]
        for round_index, forwarded in enumerate(forwarded_by_round, start=1):
            hooks.note_round(forwarded=forwarded)
    finally:
        gc.callbacks.remove(record)

    # The 3rd and 6th forwarded rounds; rounds without a forward do not count.
    assert full_collections == [4, 8]


def test_cycles_dropped_in_gen2_are_freed_by_the_hooks(serving_collector) -> None:
    interval = 4
    hooks = GcHooks(full_collection_interval=interval)
    live: weakref.WeakSet = weakref.WeakSet()
    for round_index in range(1, 3 * interval + 1):
        request = _Cycle()
        live.add(request)
        _promote_to_oldest_generation()
        # Dropped in generation 2: no young collection frees it, and CPython
        # starts no full collection by itself while serving.
        del request
        hooks.note_round(forwarded=True)
        # The hooks bound what is left to the rounds since their last
        # collection.
        assert len(live) == round_index % interval


def test_without_the_hooks_cycles_stay_until_a_manual_collection(
    serving_collector,
) -> None:
    live: weakref.WeakSet = weakref.WeakSet()
    for _ in range(8):
        request = _Cycle()
        live.add(request)
        _promote_to_oldest_generation()
        del request
    # Nothing frees them on its own, which is why the loop runs the hooks...
    assert len(live) == 8
    # ...and a manual full collection still does.
    assert gc.collect() >= 8
    assert len(live) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
