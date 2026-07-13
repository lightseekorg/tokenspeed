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

"""CPU-only tests for the V4 flat-arena sleep/wake generation handshake."""

from __future__ import annotations

import ast
import pathlib
from types import SimpleNamespace

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_EVENT_LOOP = _ROOT / "python" / "tokenspeed" / "runtime" / "engine" / "event_loop.py"
_METHOD_NAMES = {
    "_kv_pools",
    "_v4_flat_arenas",
    "_reset_caches_for_release",
    "_kv_repair_after_wake",
}


def _load_event_loop_seam():
    """Compile only the dependency-free lifecycle methods from EventLoop."""
    tree = ast.parse(_EVENT_LOOP.read_text())
    source_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EventLoop"
    )
    methods = [
        node
        for node in source_class.body
        if isinstance(node, ast.FunctionDef) and node.name in _METHOD_NAMES
    ]
    assert {node.name for node in methods} == _METHOD_NAMES
    seam_class = ast.ClassDef(
        name="EventLoopSeam",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                seam_class,
            ],
            type_ignores=[],
        )
    )
    namespace = {}
    exec(compile(module, str(_EVENT_LOOP), "exec"), namespace)
    return namespace["EventLoopSeam"]


EventLoopSeam = _load_event_loop_seam()


class _Arena:
    def __init__(self, returned_generation: int | None = None):
        self.returned_generation = returned_generation
        self.expected_generations = []

    def repair_after_wake(self, *, expected_generation: int) -> int:
        self.expected_generations.append(expected_generation)
        if self.returned_generation is not None:
            return self.returned_generation
        return expected_generation


class _Pool:
    def __init__(self, arena=None):
        self.flat_arena_set = arena
        self.clear_calls = 0

    def clear_kv_buffers(self):
        self.clear_calls += 1


class _Scheduler:
    def __init__(self, generation=1):
        self.generation = generation
        self.flat_reset_calls = 0
        self.prefix_reset_calls = 0

    def reset_flat_kv_cache(self):
        self.flat_reset_calls += 1
        return self.generation

    def reset_prefix_cache(self):
        self.prefix_reset_calls += 1


def _event_loop(*, target, draft=None, scheduler=None):
    event_loop = EventLoopSeam()
    event_loop.model_executor = SimpleNamespace(
        token_to_kv_pool=target,
        draft_token_to_kv_pool=draft,
    )
    event_loop.scheduler = scheduler or _Scheduler()
    event_loop._flat_kv_release_generation = None
    return event_loop


def test_shared_target_draft_arena_resets_and_repairs_once():
    arena = _Arena()
    target = _Pool(arena)
    draft = _Pool(arena)
    scheduler = _Scheduler(generation=1)
    event_loop = _event_loop(target=target, draft=draft, scheduler=scheduler)

    event_loop._reset_caches_for_release()

    assert scheduler.flat_reset_calls == 1
    assert scheduler.prefix_reset_calls == 0
    assert event_loop._flat_kv_release_generation == 1

    event_loop._kv_repair_after_wake()

    assert arena.expected_generations == [1]
    assert target.clear_calls == 0
    assert draft.clear_calls == 0
    assert event_loop._flat_kv_release_generation is None


def test_flat_wake_requires_a_matching_scheduler_reset_generation():
    arena = _Arena()
    event_loop = _event_loop(target=_Pool(arena))

    with pytest.raises(RuntimeError, match="no matching scheduler reset generation"):
        event_loop._kv_repair_after_wake()

    assert arena.expected_generations == []


def test_generation_mismatch_keeps_release_pending_for_diagnosis():
    arena = _Arena(returned_generation=8)
    event_loop = _event_loop(target=_Pool(arena), scheduler=_Scheduler(generation=7))
    event_loop._reset_caches_for_release()

    with pytest.raises(RuntimeError, match="unexpected generation"):
        event_loop._kv_repair_after_wake()

    assert arena.expected_generations == [7]
    assert event_loop._flat_kv_release_generation == 7


def test_distinct_target_and_draft_arenas_are_rejected_before_reset():
    scheduler = _Scheduler()
    event_loop = _event_loop(
        target=_Pool(_Arena()),
        draft=_Pool(_Arena()),
        scheduler=scheduler,
    )

    with pytest.raises(RuntimeError, match="must share one canonical arena"):
        event_loop._reset_caches_for_release()

    assert scheduler.flat_reset_calls == 0
    assert scheduler.prefix_reset_calls == 0


@pytest.mark.parametrize("generation", [True, 0, -1, "1", None])
def test_flat_reset_rejects_invalid_scheduler_generation(generation):
    event_loop = _event_loop(
        target=_Pool(_Arena()), scheduler=_Scheduler(generation=generation)
    )

    with pytest.raises(RuntimeError, match="invalid generation"):
        event_loop._reset_caches_for_release()

    assert event_loop._flat_kv_release_generation is None


def test_radix_wake_retains_legacy_per_pool_repair():
    target = _Pool()
    draft = _Pool()
    scheduler = _Scheduler()
    event_loop = _event_loop(target=target, draft=draft, scheduler=scheduler)

    event_loop._reset_caches_for_release()
    event_loop._kv_repair_after_wake()

    assert scheduler.prefix_reset_calls == 1
    assert scheduler.flat_reset_calls == 0
    assert target.clear_calls == 1
    assert draft.clear_calls == 1
