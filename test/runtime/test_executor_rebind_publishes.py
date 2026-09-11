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

"""The rebind's own publish and rebuild bodies, run rather than stubbed.

``test_executor_rebind_step`` pins the order the rebind calls things in, with
``_configure_for_pools`` and ``_build_graph_owners`` replaced by recorders --
so it cannot see what those two actually do. These run them: the layer stamps
and the backends' ``configure_runtime`` have to be re-issued against the
replacement pool, and both graph owners have to be new objects built for it.
The observable property is "rebound == freshly built": the owners a rebind
leaves behind must carry exactly the pools a first build would have given
them.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
import pytest  # noqa: E402
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.execution import model_executor as module  # noqa: E402
from tokenspeed.runtime.execution.model_executor import ModelExecutor  # noqa: E402


def _pool(tag, *, family="history"):
    return SimpleNamespace(
        arena=SimpleNamespace(
            runtime_contract=SimpleNamespace(
                tag=f"{tag} contract",
                group_specs=(SimpleNamespace(family=family),),
            ),
            cache_group_specs=(f"{tag} spec",),
            cache_group_page_counts={f"{tag} group": 3},
        )
    )


class _Backend:
    cache_consumer_families = ("history",)

    def __init__(self, name, log):
        self.name = name
        self._log = log
        self.pool = None

    def set_cache_pool(self, pool):
        self.pool = pool
        self._log.append((self.name, "set_cache_pool", pool))

    def configure_runtime(self, *, cache_group_specs, cache_group_page_counts):
        self._log.append(
            (self.name, "configure_runtime", cache_group_specs, cache_group_page_counts)
        )


class _Drafter:
    """Stands in for a drafter; the hook is where cached views get rebuilt."""

    def __init__(self, token_to_kv_pool, log):
        self.draft_model_runner = SimpleNamespace(model="draft model")
        self.token_to_kv_pool = token_to_kv_pool
        self._log = log

    def set_cache_pool(self, token_to_kv_pool):
        self.token_to_kv_pool = token_to_kv_pool
        self._log.append(("drafter", "set_cache_pool", token_to_kv_pool))


class _Owner:
    """Stands in for a graph owner; records the pools it was built for."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def release_graphs(self):
        self.kwargs.setdefault("released", 0)
        self.kwargs["released"] += 1


def _executor(monkeypatch, log, *, with_draft: bool):
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.device = "cuda"
    executor.config = SimpleNamespace(spec_algo=None, enforce_eager=False)
    executor.model_runner = SimpleNamespace(model="target model")
    executor.attn_backend = _Backend("target", log)
    executor.token_to_kv_pool = _pool("probe")
    executor._cache_runtime_contract = "probe contract"
    executor.input_buffers = "input buffers"
    executor.sampling_backend = "sampling"
    executor.runtime_states = "runtime states"
    executor.grammar_runtime = None
    executor._graph_support = SimpleNamespace(decode_graph=True, prefill_graph=True)
    if with_draft:
        executor.draft_attn_backend = _Backend("draft", log)
        executor.draft_token_to_kv_pool = _pool("probe draft")
        executor.drafter = _Drafter(executor.draft_token_to_kv_pool, log)
        executor._draft_model_runner = executor.drafter.draft_model_runner
    else:
        executor.draft_attn_backend = None
        executor.draft_token_to_kv_pool = None
        executor.drafter = None
        executor._draft_model_runner = None

    monkeypatch.setattr(
        module,
        "bind_cache_groups",
        lambda model, pool: log.append(("bind_cache_groups", model, pool)),
    )
    monkeypatch.setattr(
        module, "workspace_pool", lambda device: SimpleNamespace(unfreeze=lambda: None)
    )
    monkeypatch.setattr(module, "ForwardStepRunner", _Owner)
    monkeypatch.setattr(module, "PrefillGraph", _Owner)
    executor._build_graph_owners()
    log.clear()
    return executor


def test_the_rebind_restamps_both_models_onto_the_new_pool(monkeypatch):
    """The layer stamps are the caller's to re-publish, on every rebind.

    The backend tree rebinds itself; the per-layer cache-group stamps
    ``bind_cache_groups`` writes on the model do not travel with it. A stamp
    written once at construction points every attention layer at the probe
    arena's groups for the rest of the process.
    """
    log = []
    executor = _executor(monkeypatch, log, with_draft=True)
    target, draft = _pool("real"), _pool("real draft")

    executor.set_cache_pool(target, draft)

    stamps = [entry for entry in log if entry[0] == "bind_cache_groups"]
    assert stamps == [
        ("bind_cache_groups", "target model", target),
        ("bind_cache_groups", "draft model", draft),
    ]


def test_the_rebind_reconfigures_both_backends_with_the_new_specs(monkeypatch):
    """``configure_runtime`` ran before the executor was returned, so a rebind
    re-runs it -- with the replacement pool's specs and page counts, never the
    probe's."""
    log = []
    executor = _executor(monkeypatch, log, with_draft=True)
    target, draft = _pool("real"), _pool("real draft")

    executor.set_cache_pool(target, draft)

    configured = [entry for entry in log if entry[1] == "configure_runtime"]
    assert configured == [
        ("target", "configure_runtime", ("real spec",), {"real group": 3}),
        (
            "draft",
            "configure_runtime",
            ("real draft spec",),
            {"real draft group": 3},
        ),
    ]


def test_the_rebind_builds_new_graph_owners_for_the_new_pools(monkeypatch):
    """A graph owner is built for one pool; a rebind replaces it.

    Both owners cache the pool they were handed (the decode runner also sizes
    its placeholder tables from the arena), so keeping either across a rebind
    keeps the probe arena alive inside the serving path.
    """
    log = []
    executor = _executor(monkeypatch, log, with_draft=True)
    before = (executor.forward_step, executor.prefill_graph)
    target, draft = _pool("real"), _pool("real draft")

    executor.set_cache_pool(target, draft)

    assert executor.forward_step is not before[0]
    assert executor.prefill_graph is not before[1]
    assert executor.forward_step.kwargs["token_to_kv_pool"] is target
    assert executor.forward_step.kwargs["draft_token_to_kv_pool"] is draft
    assert executor.prefill_graph.kwargs["token_to_kv_pool"] is target


def test_a_rebind_to_an_unconsumable_pool_fails_where_it_happens(monkeypatch):
    """The pool that serves is the one that has to satisfy the backend.

    A family with no consumer means that group's tables are never read -- a
    capture-path assert at best, wrong pages at worst -- so the check belongs
    on the publish path both binds go through, not only on construction.
    """
    executor = _executor(monkeypatch, [], with_draft=False)

    with pytest.raises(RuntimeError, match="missing"):
        executor.set_cache_pool(_pool("real", family="state"), None)


def test_a_second_rebind_restamps_again(monkeypatch):
    """Nothing may latch the first publish: a probe is one rebind, not the last."""
    log = []
    executor = _executor(monkeypatch, log, with_draft=True)
    first, first_draft = _pool("first"), _pool("first draft")
    second, second_draft = _pool("second"), _pool("second draft")

    executor.set_cache_pool(first, first_draft)
    executor.set_cache_pool(second, second_draft)

    stamps = [entry for entry in log if entry[0] == "bind_cache_groups"]
    assert stamps[-2:] == [
        ("bind_cache_groups", "target model", second),
        ("bind_cache_groups", "draft model", second_draft),
    ]
    assert executor.forward_step.kwargs["token_to_kv_pool"] is second


def test_a_draftless_rebind_stamps_only_the_target(monkeypatch):
    log = []
    executor = _executor(monkeypatch, log, with_draft=False)
    target = _pool("real")

    executor.set_cache_pool(target, None)

    stamps = [entry for entry in log if entry[0] == "bind_cache_groups"]
    assert stamps == [("bind_cache_groups", "target model", target)]
    assert executor.forward_step.kwargs["draft_token_to_kv_pool"] is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
