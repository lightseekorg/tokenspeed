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

"""The executor's rebind step: what it releases, republishes and rebuilds."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

# Executed as a script by run_ci_suite: the test dir must be importable.
_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
import pytest  # noqa: E402
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.execution import model_executor as module  # noqa: E402
from tokenspeed.runtime.execution.model_executor import ModelExecutor  # noqa: E402


def _pool(contract):
    return SimpleNamespace(arena=SimpleNamespace(runtime_contract=contract))


class _FakeDrafter:
    """Stands in for the real hook: a drafter that caches views rebuilds them."""

    def __init__(self, token_to_kv_pool, order):
        self.token_to_kv_pool = token_to_kv_pool
        self._order = order

    def set_cache_pool(self, token_to_kv_pool):
        self.token_to_kv_pool = token_to_kv_pool
        self._order.append("drafter adopt")


def _executor(order):
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.device = "cuda"
    executor.forward_step = SimpleNamespace(
        release_graphs=lambda: order.append("release decode")
    )
    executor.prefill_graph = SimpleNamespace(
        release_graphs=lambda: order.append("release prefill")
    )
    executor.token_to_kv_pool = _pool("old")
    executor.draft_token_to_kv_pool = _pool("old draft")
    executor._cache_runtime_contract = "old"
    executor.drafter = _FakeDrafter(executor.draft_token_to_kv_pool, order)
    executor._configure_for_pools = lambda: order.append("configure")
    executor._build_graph_owners = lambda: order.append("build owners")
    return executor


def test_capture_graphs_hands_both_owners_the_entries_and_the_observer(monkeypatch):
    """Both ladders are sampled the same way, or the projection is not a sample.

    A prefill owner left capturing the whole ladder turns a four-graph probe
    into a full serving capture against the probe arena; a decode owner given
    no observer leaves the projection two samples short and raises at boot.
    """
    seen = []
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.device = "cuda"
    executor.forward_step = SimpleNamespace(
        disable=False,
        capture=lambda *, entries, observer: seen.append(("decode", entries, observer)),
    )
    executor.prefill_graph = SimpleNamespace(
        disable=False,
        capture=lambda wrapper, *, entries, observer: seen.append(
            ("prefill", entries, observer, wrapper)
        ),
    )
    monkeypatch.setattr(
        module, "workspace_pool", lambda device: SimpleNamespace(freeze=lambda: None)
    )
    observer = object()

    executor.capture_graphs(entries=3, observer=observer)

    assert seen == [
        ("decode", 3, observer),
        ("prefill", 3, observer, executor.forward_step),
    ]


def test_the_base_drafter_hook_takes_the_replacement_pool():
    """The hook has to do something; every rebind test so far stubs it out."""
    from tokenspeed.runtime.execution.drafter.base import BaseDrafter

    drafter = BaseDrafter.__new__(BaseDrafter)
    drafter.token_to_kv_pool = "probe"

    BaseDrafter.set_cache_pool(drafter, "real")

    assert drafter.token_to_kv_pool == "real"


def test_the_dflash_hook_restacks_its_views_and_drops_its_pointer_cache():
    """DFlash pre-stacks raw KV views, so taking the pool is not enough.

    Left alone, the stacked views and the module-level pointer cache keep
    naming the released probe arena: the draft then scatters into freed
    memory, which shows up as wrong tokens or an IMA at serving, never at
    boot.
    """
    from tokenspeed.runtime.execution.drafter import _dflash_fused_kv
    from tokenspeed.runtime.execution.drafter.dflash import DFlash

    _dflash_fused_kv._cached_kv_ptrs[1234] = ("stale k", "stale v")
    drafter = DFlash.__new__(DFlash)
    drafter.token_to_kv_pool = "probe"
    restacked = []
    drafter._init_fused_kv_helper = lambda: restacked.append(drafter.token_to_kv_pool)

    drafter.set_cache_pool("real")

    assert drafter.token_to_kv_pool == "real"
    assert restacked == ["real"]
    assert _dflash_fused_kv._cached_kv_ptrs == {}


def test_releasing_frees_both_owners_and_the_workspace(monkeypatch):
    """A captured graph's pool is not returned by empty_cache.

    The caller releases before it profiles device memory for the replacement
    arena, so what the probe captured is not counted as spent. The collection
    is part of that: the graphs sit in reference cycles, and the profile's own
    empty_cache cannot return what a cycle still holds.
    """
    order = []
    executor = _executor(order)
    monkeypatch.setattr(
        module,
        "workspace_pool",
        lambda device: SimpleNamespace(
            unfreeze=lambda: order.append(("unfreeze", device))
        ),
    )
    monkeypatch.setattr(module.gc, "collect", lambda: order.append("collect"))

    executor.release_graphs()

    assert order == [
        "release decode",
        "release prefill",
        ("unfreeze", "cuda"),
        "collect",
    ]


def test_adopting_republishes_every_executor_side_reference():
    """The backends took the pool when it was built; these did not."""
    order = []
    executor = _executor(order)
    target, draft = _pool("new"), _pool("new draft")

    executor.set_cache_pool(target, draft)

    assert executor.token_to_kv_pool is target
    assert executor.draft_token_to_kv_pool is draft
    assert executor._cache_runtime_contract == "new"
    assert executor.drafter.token_to_kv_pool is draft
    assert order == ["drafter adopt", "configure", "build owners"]


def test_adopting_does_not_bind_the_backends():
    """One owner for the bind: the factory that built the pool.

    A second set_cache_pool would drop the verify workspace that factory
    allocated for the replacement pool moments earlier.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(ModelExecutor.set_cache_pool)))
    receivers = {
        node.func.value.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "set_cache_pool"
        and isinstance(node.func.value, ast.Attribute)
    }

    # The drafter re-stacks its views; the backends were bound by the factory.
    assert receivers <= {"drafter"}, receivers


def test_adopting_without_a_draft_side_carries_none():
    order = []
    executor = _executor(order)
    executor.drafter = None
    target = _pool("new")

    executor.set_cache_pool(target, None)

    assert executor.token_to_kv_pool is target
    assert executor.draft_token_to_kv_pool is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
