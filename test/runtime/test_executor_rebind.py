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

"""The executor's rebind: what it captures, releases, republishes and rebuilds."""

from __future__ import annotations

import importlib
import os
import sys
from types import SimpleNamespace

# Executed as a script by run_ci_suite: the test dir must be importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest  # noqa: E402
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.execution import model_executor as module  # noqa: E402
from tokenspeed.runtime.execution.drafter.base import BaseDrafter  # noqa: E402
from tokenspeed.runtime.execution.model_executor import ModelExecutor  # noqa: E402


def _pool(tag, *, family="history"):
    return SimpleNamespace(
        arena=SimpleNamespace(
            runtime_contract=SimpleNamespace(
                tag=tag, group_specs=(SimpleNamespace(family=family),)
            ),
            cache_group_specs=(f"{tag} spec",),
            cache_group_page_counts={f"{tag} group": 3},
        )
    )


class _Holder:
    """A backend, drafter or context producer; logs what it is handed."""

    cache_consumer_families = ("history",)

    def __init__(self, name, log, pool=None):
        self.name, self.log, self.token_to_kv_pool = name, log, pool
        self.draft_model_runner = SimpleNamespace(model="draft model")

    def set_cache_pool(self, pool):
        self.token_to_kv_pool = pool
        self.log.append((self.name, "set_cache_pool", pool))

    def configure_runtime(self, *, cache_group_specs, cache_group_page_counts):
        self.log.append(
            (self.name, "configure", cache_group_specs, cache_group_page_counts)
        )


def _executor(monkeypatch, log, *, with_draft: bool):
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.device = "cuda"
    executor.config = SimpleNamespace(spec_algo=None, enforce_eager=False)
    executor.model_runner = SimpleNamespace(model="target model")
    executor.attn_backend = _Holder("target", log)
    executor.token_to_kv_pool = _pool("probe")
    executor._cache_runtime_contract = "probe contract"
    executor.input_buffers = "input buffers"
    executor.sampling_backend = "sampling"
    executor.runtime_states = "runtime states"
    executor.grammar_runtime = None
    executor._graph_support = SimpleNamespace(decode_graph=True, prefill_graph=True)
    draft = _pool("probe draft") if with_draft else None
    executor.draft_token_to_kv_pool = draft
    executor.draft_attn_backend = with_draft and _Holder("draft", log) or None
    executor.drafter = with_draft and _Holder("drafter", log, draft) or None
    executor._draft_model_runner = (
        with_draft and executor.drafter.draft_model_runner or None
    )
    executor.dspark_context_producer = (
        with_draft and _Holder("producer", log, draft) or None
    )

    def owner(**kwargs):
        log.append(("owner",))
        return SimpleNamespace(kwargs=kwargs, **kwargs)

    validate = module.validate_scheduler_config
    monkeypatch.setattr(
        module,
        "validate_scheduler_config",
        lambda **kw: log.append(("validate", kw["attn_backend"], kw["kv_pool"]))
        or validate(**kw),
    )
    monkeypatch.setattr(
        module,
        "bind_cache_groups",
        lambda model, pool: log.append(("bind", model, pool)),
    )
    monkeypatch.setattr(module, "ForwardStepRunner", owner)
    monkeypatch.setattr(module, "PrefillGraph", owner)
    executor._build_graph_owners()
    log.clear()
    return executor


@pytest.mark.parametrize("with_draft", [True, False])
def test_a_rebind_republishes_every_holder_onto_the_new_pools(monkeypatch, with_draft):
    log = []
    executor = _executor(monkeypatch, log, with_draft=with_draft)
    target = _pool("real")
    draft = _pool("real draft") if with_draft else None

    # Nothing may latch the first publish: the probe is one rebind, not the last.
    executor.set_cache_pool(_pool("first"), draft and _pool("first draft"))
    log.clear()
    executor.set_cache_pool(target, draft)

    expected = [("validate", executor.attn_backend, target)]
    if with_draft:
        expected += [
            ("drafter", "set_cache_pool", draft),
            ("producer", "set_cache_pool", draft),
        ]
    expected += [("target", "configure", ("real spec",), {"real group": 3})]
    if with_draft:
        expected += [
            ("draft", "configure", ("real draft spec",), {"real draft group": 3})
        ]
    expected += [("bind", "target model", target)]
    if with_draft:
        expected += [("bind", "draft model", draft)]
    # The backends took the pool when it was built; binding again drops their workspace.
    assert log == expected + [("owner",), ("owner",)]
    assert executor._cache_runtime_contract is target.arena.runtime_contract
    assert executor.forward_step.token_to_kv_pool is target
    assert executor.forward_step.draft_token_to_kv_pool is draft
    assert executor.prefill_graph.token_to_kv_pool is target

    # No holder of a pool is left on the replaced arena.
    for name, value in vars(executor).items():
        for held in (
            value,
            *(vars(value).values() if hasattr(value, "__dict__") else ()),
        ):
            if getattr(held, "arena", None) is not None:
                assert held is target or held is draft, name


def test_a_rebind_to_an_unconsumable_pool_fails_where_it_happens(monkeypatch):
    executor = _executor(monkeypatch, [], with_draft=False)
    with pytest.raises(RuntimeError, match="missing"):
        executor.set_cache_pool(_pool("real", family="state"), None)


@pytest.mark.parametrize("spec_algo", ["DFLASH", "DSPARK"])
def test_a_block_drafter_is_rechecked_against_the_targets_new_pool(
    monkeypatch, spec_algo
):
    log = []
    executor = _executor(monkeypatch, log, with_draft=True)
    executor.config = SimpleNamespace(spec_algo=spec_algo, enforce_eager=False)
    monkeypatch.setattr(
        module, "check_block_drafter_storage", lambda m, p: log.append(("check", m, p))
    )
    target = _pool("real")

    executor.set_cache_pool(target, _pool("real draft"))

    assert [e for e in log if e[0] == "check"] == [("check", "draft model", target)]


def test_capture_hands_both_owners_the_entries_and_the_drafter_its_window(
    monkeypatch,
):
    seen = []
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.device = "cuda"
    executor.forward_step = SimpleNamespace(
        disable=False,
        stream="stream",
        capture=lambda *, entries, observer: seen.append(("decode", entries, observer)),
    )
    executor.prefill_graph = SimpleNamespace(
        disable=False,
        capture=lambda wrapper, *, entries, observer: seen.append(
            ("prefill", entries, observer, wrapper)
        ),
    )
    executor.drafter = SimpleNamespace(
        captures_prefill_graph=True,
        capture_prefill_graph=lambda stream, window: seen.append(
            ("drafter", stream, window)
        ),
    )
    monkeypatch.setattr(
        module, "workspace_pool", lambda _d: SimpleNamespace(freeze=lambda: None)
    )
    observer = SimpleNamespace(measure=lambda series: f"window:{series}")

    executor.capture_graphs(entries=3, observer=observer)

    assert seen == [
        ("decode", 3, observer),
        ("prefill", 3, observer, executor.forward_step),
        ("drafter", "stream", "window:prefill:drafter"),
    ]


def test_releasing_frees_both_owners_the_drafter_and_the_workspace(monkeypatch):
    order = []
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.device = "cuda"
    executor.forward_step = SimpleNamespace(
        release_graphs=lambda: order.append("decode")
    )
    executor.prefill_graph = SimpleNamespace(
        release_graphs=lambda: order.append("prefill")
    )
    executor.drafter = SimpleNamespace(
        release_prefill_graph=lambda: order.append("drafter")
    )
    monkeypatch.setattr(
        module,
        "workspace_pool",
        lambda device: SimpleNamespace(
            unfreeze=lambda: order.append(("unfreeze", device))
        ),
    )
    monkeypatch.setattr(module.gc, "collect", lambda: order.append("collect"))

    executor.release_graphs()

    assert order == ["decode", "prefill", "drafter", ("unfreeze", "cuda"), "collect"]


def test_the_drafter_hooks_take_the_pool_and_drop_what_named_the_old_one(monkeypatch):
    from tokenspeed.runtime.execution.drafter import _dflash_fused_kv
    from tokenspeed.runtime.execution.drafter.deepseek_v4_dspark import DeepseekV4DSpark
    from tokenspeed.runtime.execution.drafter.dflash import DFlash
    from tokenspeed.runtime.execution.dspark_context import DSparkContextProducer

    base = BaseDrafter.__new__(BaseDrafter)
    BaseDrafter.set_cache_pool(base, "real")
    assert base.token_to_kv_pool == "real"

    # DFlash pre-stacks raw KV views and caches their pointers module-wide.
    monkeypatch.setitem(_dflash_fused_kv._cached_kv_ptrs, 1234, ("k", "v"))
    dflash = DFlash.__new__(DFlash)
    rebuilt = []
    dflash._init_fused_kv_helper = lambda: rebuilt.append("fused_kv")
    dflash._init_incremental_proj = lambda: rebuilt.append("incremental_proj")
    dflash.set_cache_pool("real")
    assert dflash.token_to_kv_pool == "real"
    assert _dflash_fused_kv._cached_kv_ptrs == {}
    assert rebuilt == ["fused_kv", "incremental_proj"]

    dspark = DeepseekV4DSpark.__new__(DeepseekV4DSpark)
    dspark._prefill_graph = object()
    dspark.release_prefill_graph()
    assert dspark._prefill_graph is None

    # Only the last pipeline stage owns the draft context cache.
    producer = DSparkContextProducer.__new__(DSparkContextProducer)
    producer.token_to_kv_pool = _pool("probe draft")
    producer.set_cache_pool(_pool("real draft"))
    with pytest.raises(ValueError, match="ownership"):
        producer.set_cache_pool(None)


def test_a_drafter_that_captures_a_prefill_graph_declares_and_releases_it():
    for name in (
        "deepseek_v41_dspark",
        "deepseek_v4_dspark",
        "dflash",
        "dflash2",
        "dspark",
        "eagle",
        "mtp",
    ):
        importlib.import_module(f"tokenspeed.runtime.execution.drafter.{name}")

    def _descendants(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from _descendants(sub)

    concrete = list(_descendants(BaseDrafter))
    assert len(concrete) >= 7
    hooks = {"capture_prefill_graph", "captures_prefill_graph", "release_prefill_graph"}
    for cls in concrete:
        assert hooks & set(vars(cls)) in (set(), hooks), cls.__name__


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
