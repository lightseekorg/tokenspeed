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

"""The probe arena: its size, which boots get one, and how the factory builds it."""

from __future__ import annotations

import ast
import contextlib
import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch

# Both entry points need the test dir (siblings) and the root (test.runtime.*).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.execution import device  # noqa: E402
from tokenspeed.runtime.execution.cudagraph_memory import (  # noqa: E402
    probe_arena_parent_blocks,
)
from tokenspeed.runtime.layers.attention import registry  # noqa: E402
from tokenspeed.runtime.layers.attention.kv_cache.recipes import setup  # noqa: E402

RUNTIME = pathlib.Path(__file__).resolve().parents[2] / "python/tokenspeed/runtime"


def _kimi_k3_recipe(**kwargs):
    from test.runtime.conftest import kimi_recipe

    return kimi_recipe(**{"max_bs": 8, **kwargs})


def _glm53_flash_recipe():
    from test.runtime.test_glm53_flash_cache_spec import _recipe

    return _recipe(tp_size=8, mla_cache_dtype=torch.bfloat16)


def _deepseek_v4_recipe():
    from test.runtime.test_deepseek_v4_config import _v4_recipe

    return _v4_recipe(
        SimpleNamespace(
            num_hidden_layers=3,
            compress_ratios=[0, 4, 128],
            num_attention_heads=64,
            head_dim=512,
            qk_rope_head_dim=64,
            sliding_window=128,
            index_head_dim=128,
            attention_config={},
        )
    )


def _deepseek_v41_recipe():
    from test.runtime.test_deepseek_v41_cache import _recipe

    return _recipe("cpu")


# The families whose per-group demand sizes their own parents.
_RECIPES = {
    "kimi_k3": _kimi_k3_recipe,
    "glm53_flash": _glm53_flash_recipe,
    "deepseek_v4": _deepseek_v4_recipe,
    "deepseek_v41": _deepseek_v41_recipe,
}
_NO_FIXTURE = {
    "mha",
    "mla",
    "dsa",
    "msa",
    "qwen_gdn",
    "qwen4_exp",
    "inkling",
    "mamba2",
}


def _function(module: str, name: str) -> ast.FunctionDef:
    tree = ast.parse((RUNTIME / module).read_text())
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _keyword(function: ast.FunctionDef, callee: str, keyword: str) -> ast.expr:
    call = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", None)) == callee
        and any(kw.arg == keyword for kw in node.keywords)
    )
    return next(kw.value for kw in call.keywords if kw.arg == keyword)


def test_every_registered_family_is_either_driven_here_or_named() -> None:
    assert set(setup._RECIPES) == set(_RECIPES) | _NO_FIXTURE


@pytest.mark.parametrize("rows", [2, 1024])
@pytest.mark.parametrize("family", sorted(_RECIPES))
def test_every_family_binds_a_probe_arena_and_reports_its_cost(family, rows) -> None:
    recipe = _RECIPES[family]()
    recipe.cache_budget_bytes = 0
    recipe.probe_batch_rows = rows

    built = recipe.setup()

    plan = built.spec.memory_plan
    assert built.spec.token_capacity > 0
    # A floor, not a size: the arena also has to admit one token per group.
    assert plan.num_lcm_blocks >= rows
    assert built.cache_budget_bytes == recipe.workspace_bytes() + plan.arena_bytes
    if family == "kimi_k3" and rows == 1024:
        assert plan.num_lcm_blocks == 1024


def test_the_probe_arena_does_not_grow_with_max_num_seqs() -> None:
    def arena_bytes(max_bs: int) -> int:
        recipe = _kimi_k3_recipe(max_bs=max_bs)
        recipe.cache_budget_bytes = 0
        recipe.probe_batch_rows = 2
        return recipe.setup().spec.memory_plan.arena_bytes

    assert arena_bytes(8) == arena_bytes(512)
    # A cap, not a floor: a server smaller than the probe's batch stays smaller.
    assert arena_bytes(1) < arena_bytes(512)


def test_the_override_reaches_the_plan_through_the_factory_seam() -> None:
    template = _kimi_k3_recipe()
    built = setup.prepare_cache_setup(
        family="kimi_k3",
        server_args=template.server_args,
        model_config=template.model_config,
        attn_config=template.attn_config,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=0,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
        probe_batch_rows=64,
    )

    plan = built.spec.memory_plan
    assert plan.num_lcm_blocks >= 64
    assert built.cache_budget_bytes == template.workspace_bytes() + plan.arena_bytes


@pytest.mark.parametrize(
    "tokens, context, capture_batch_sizes, expected",
    [
        (8192, 4096, None, 2),
        (32768, 8192, None, 4),
        (8192, 3000, None, 3),
        (8193, 4096, None, 3),
        (100, 64, None, 2),
        # A configured capture batch size fabricates that many rows whatever the bucket.
        (8192, 40960, None, 1),
        (8192, 40960, [1, 8, 32], 32),
    ],
)
def test_the_floor_is_the_widest_row_count_the_probe_fabricates(
    tokens, context, capture_batch_sizes, expected
) -> None:
    from tokenspeed.runtime.execution.prefill_graph import get_prefill_token_buckets

    floor = probe_arena_parent_blocks(
        max_forward_tokens=tokens,
        context_len=context,
        capture_batch_sizes=capture_batch_sizes,
        decoder_requests=0,
    )
    assert floor == expected
    widest = max(
        get_prefill_token_buckets(
            SimpleNamespace(
                disable_prefill_graph=False,
                prefill_graph_max_tokens=tokens,
                prefill_graph_capture_sizes=None,
                chunked_prefill_size=tokens,
            )
        )
    )
    assert floor * context >= widest


@pytest.mark.parametrize(
    "ceiling, capture_batch_sizes, context, rows_per_request, max_num_seqs, expected",
    [
        (None, None, 4096, None, 80, 2),
        # A graph ladder wider than the per-forward token budget.
        (65536, None, 4096, None, 80, 16),
        (None, [1, 8, 32], 40960, None, 80, 32),
        # A narrowing decoder ladder fabricates a request per 128 rows, up to max_num_seqs.
        (None, None, 40960, 128, 80, 64),
        (None, None, 40960, 128, 16, 16),
    ],
)
def test_the_boot_floor_reads_both_knobs(
    ceiling, capture_batch_sizes, context, rows_per_request, max_num_seqs, expected
) -> None:
    server_args = SimpleNamespace(
        all2all_backend="none",
        prefill_graph_max_tokens=ceiling,
        chunked_prefill_size=8192,
        max_total_tokens=None,
        prefill_graph_capture_batch_sizes=capture_batch_sizes,
        max_num_seqs=max_num_seqs,
    )
    floor = device.probe_arena_floor(
        server_args,
        SimpleNamespace(context_len=context),
        8192,
        decoder_rows_per_request=rows_per_request,
    )
    assert floor == expected


def _callee(node: ast.Call) -> str | None:
    return getattr(node.func, "id", getattr(node.func, "attr", None))


class _ServingCapture(Exception):
    """Ends a fabricated boot where the serving capture would begin."""


def _startup_charge(monkeypatch, *, init_keeps, tune_keeps, enforce_eager, is_nvidia):
    """Boot with fakes; returns (startup bytes each rebind got, devices read, stack kept)."""
    from tokenspeed.runtime.execution import factory

    free, reads, charged, stack = [1 << 40], [], [], [0]

    class Executor:
        attn_backend = draft_attn_backend = None

        def __init__(self):
            free[0] -= init_keeps

        def autotune(self):
            # Tried tactics also raise the stack limit, which reserves local memory.
            free[0] -= tune_keeps + (1 << 30)
            stack[0] += 1 << 30

        def capture_graphs(self, *, entries, observer):
            raise _ServingCapture

    def build(*args, **kwargs):
        # The probe build and the rebind take memory too: neither is startup's.
        free[0] -= 192 << 20
        return built

    def rebind(executor, build, args, gpu, probe, backends, *, startup_resident_bytes):
        charged.append(startup_resident_bytes)
        free[0] -= 1 << 30
        return probe, views(probe)

    def views(attention):
        return SimpleNamespace(
            token_to_kv_pool=None,
            draft_token_to_kv_pool=None,
            cache_geometry=SimpleNamespace(prefix_granularity=1),
            cache_groups=[],
        )

    target = SimpleNamespace(
        model=SimpleNamespace(),
        prepare_multimodal_runtime=lambda: None,
        prepare_communication_runtime=lambda _tokens: None,
    )
    built = SimpleNamespace(
        attn_backend=None,
        draft_attn_backend=None,
        token_to_kv_pool=None,
        draft_token_to_kv_pool=None,
    )

    @contextlib.contextmanager
    def restore_stack_limit():
        found = stack[0]
        yield
        free[0] += stack[0] - found
        stack[0] = found

    driver = SimpleNamespace(
        synchronize=lambda gpu: reads.append(gpu),
        empty_cache=lambda: None,
        mem_get_info=lambda gpu: reads.append(gpu) or (free[0], 1 << 40),
    )
    monkeypatch.setattr(torch, "get_device_module", lambda _device: driver)
    monkeypatch.setattr(
        device,
        "current_platform",
        lambda: SimpleNamespace(
            restore_stack_limit=restore_stack_limit, is_nvidia=is_nvidia
        ),
    )
    monkeypatch.setattr(factory, "create_model_runner", lambda *a: (target, None))
    monkeypatch.setattr(factory, "create_model_executor", lambda **_: Executor())
    monkeypatch.setattr(
        factory, "ModelExecutorConfig", SimpleNamespace(from_server_args=lambda **_: 0)
    )
    monkeypatch.setattr(registry, "create_attn_components", build)
    monkeypatch.setattr(device, "probe_arena_floor", lambda *_, **__: 1)
    monkeypatch.setattr(device, "pool_views", views)
    monkeypatch.setattr(device, "_rebind_under_reserve", rebind)
    server_args = SimpleNamespace(
        disaggregation_mode="null",
        chunked_prefill_size=8192,
        attention_backend=None,
        drafter_attention_backend=None,
        disable_cudagraph_memory_reserve=False,
        enforce_eager=enforce_eager,
        enable_prefix_caching=False,
        enable_memory_saver=False,
        device="cuda",
    )
    with pytest.raises(_ServingCapture):
        device.build_device_side(
            server_args=server_args,
            model_config=SimpleNamespace(context_len=4096, is_multimodal_active=False),
            draft_model_config=None,
            gpu_id=3,
            global_rank=7,
            attn_tp_rank=1,
            min_per_gpu_mem=0,
            overlap_schedule_depth=0,
            decode_input_tokens=1,
            max_batch_size=8,
        )
    return charged, set(reads), stack[0]


@pytest.mark.parametrize(
    "init_keeps, tune_keeps", [(256 << 20, 384 << 20), (256 << 20, -512 << 20)]
)
def test_the_rebind_is_charged_what_executor_init_and_tuning_kept(
    monkeypatch, init_keeps, tune_keeps
) -> None:
    charged, devices, stack = _startup_charge(
        monkeypatch,
        init_keeps=init_keeps,
        tune_keeps=tune_keeps,
        enforce_eager=False,
        is_nvidia=True,
    )
    # The signed net of both; the probe floors it.
    assert charged == [init_keeps + tune_keeps]
    assert devices == {3} and stack == 0


def test_a_boot_without_a_probe_reads_no_startup_memory(monkeypatch) -> None:
    charged, devices, stack = _startup_charge(
        monkeypatch,
        init_keeps=1 << 30,
        tune_keeps=1 << 30,
        enforce_eager=True,
        is_nvidia=True,
    )
    assert charged == [] and devices == set() and stack == 0


def test_a_probe_off_nvidia_leaves_the_startup_residue_to_the_headroom(
    monkeypatch,
) -> None:
    charged, devices, _ = _startup_charge(
        monkeypatch,
        init_keeps=1 << 30,
        tune_keeps=1 << 30,
        enforce_eager=False,
        is_nvidia=False,
    )
    assert charged == [0] and devices == set()


def test_the_boot_probes_rebuilds_and_captures_in_order() -> None:
    build = _function("execution/device.py", "build_device_side")
    steps = {
        "_cudagraph_probe_refusal",
        "build_components",
        "autotune",
        "_rebind_under_reserve",
        "capture_graphs",
        "set_random_seed",
    }
    calls = sorted(
        (n for n in ast.walk(build) if isinstance(n, ast.Call) and _callee(n) in steps),
        key=lambda n: (n.lineno, n.col_offset),
    )
    assert [_callee(n) for n in calls] == [
        "_cudagraph_probe_refusal",
        "build_components",
        "autotune",
        "_rebind_under_reserve",
        "capture_graphs",
        "set_random_seed",
    ]
    refusal, probe_build, _, _, serving, _ = calls
    assert [ast.unparse(arg) for arg in refusal.args] == ["server_args"]
    # The target's model: the narrowing floor must not read the draft's.
    (narrowing,) = (
        n
        for n in ast.walk(build)
        if isinstance(n, ast.Call) and _callee(n) == "narrowing_prefill_model"
    )
    assert ast.unparse(narrowing.args[0]) == "target.model"
    rows = next(kw.value for kw in probe_build.keywords if kw.arg == "probe_batch_rows")
    assert isinstance(rows, ast.IfExp) and ast.unparse(rows.orelse) == "None"
    assert _callee(rows.body) == "probe_arena_floor"
    first = next(
        kw.value for kw in probe_build.keywords if kw.arg == "profiled_cache_bytes"
    )
    assert ast.unparse(first) == "None"
    # The serving capture takes the whole ladder.
    entries = next(kw.value for kw in serving.keywords if kw.arg == "entries")
    assert ast.unparse(entries) == "None"

    hops = [
        ("execution/device.py", "build_components", "create_attn_components"),
        (
            "layers/attention/registry.py",
            "create_attn_components",
            "prepare_cache_setup",
        ),
        (
            "layers/attention/kv_cache/recipes/setup.py",
            "prepare_cache_setup",
            "cache_recipe",
        ),
    ]
    for module, function, callee in hops:
        keyword = _keyword(_function(module, function), callee, "probe_batch_rows")
        assert ast.unparse(keyword) == "probe_batch_rows", (module, callee)
    forwarder = _function("execution/device.py", "build_components")
    for keyword in (
        "graph_reserve_bytes",
        "profiled_cache_bytes",
        "reuse_target_backend",
        "reuse_draft_backend",
    ):
        value = _keyword(forwarder, "create_attn_components", keyword)
        assert ast.unparse(value) == keyword
    factory = _function("layers/attention/registry.py", "create_attn_components")
    for side in ("target", "draft"):
        reused = _keyword(factory, f"_create_{side}_components", "backend")
        assert ast.unparse(reused) == f"reuse_{side}_backend"
    # One profile per boot: the rebuild reuses the probe build's, less the reserve.
    profiled = next(
        node
        for node in ast.walk(factory)
        if isinstance(node, ast.If)
        and "profile_available_cache_memory_bytes" in ast.unparse(node.body[0])
    )
    assert ast.unparse(profiled.test) == "profiled_cache_bytes is None"
    lines = [line.strip() for line in ast.unparse(factory).splitlines()]
    assert (
        "cache_memory = reserve_cache_budget(profiled_cache_bytes, "
        "graph_reserve_bytes)" in lines
    )


def test_a_probe_arena_cannot_also_carry_a_reserve() -> None:
    def _failure(reserve: int, rows: int | None) -> str:
        # The legal pairs fail later on the empty arguments, by design.
        with pytest.raises(Exception) as caught:  # noqa: PT011
            registry.create_attn_components(
                SimpleNamespace(),
                SimpleNamespace(),
                0,
                0,
                0,
                graph_reserve_bytes=reserve,
                post_profile_bytes=0,
                probe_batch_rows=rows,
                profiled_cache_bytes=None,
                reuse_target_backend=None,
                reuse_draft_backend=None,
            )
        return f"{type(caught.value).__name__}: {caught.value}"

    assert _failure(5 << 30, 100).startswith("ValueError: a probe arena")
    assert "cannot also reserve" not in _failure(0, 100)
    assert "cannot also reserve" not in _failure(5 << 30, None)


def test_pool_staged_verify_scratch_keeps_the_serving_concurrency(monkeypatch) -> None:
    from tokenspeed_kernel.ops.attention import kda as kda_ops

    from tokenspeed.runtime.layers.attention.kv_cache.recipes.base import (
        kda_verify_scratch_in_pool,
    )

    for family in sorted(_RECIPES):
        assert _RECIPES[family]().verify_scratch_in_pool() is False, family

    plain = _kimi_k3_recipe()
    speculative = _kimi_k3_recipe(
        speculative_algorithm="eagle", speculative_num_draft_tokens=2
    )
    monkeypatch.setattr(kda_ops, "kda_replay_commit_supported", lambda *a, **k: True)
    for raw_gate in (True, False):
        monkeypatch.setattr(
            kda_ops, "kda_batched_replay_uses_raw_gate", lambda *a, _r=raw_gate, **k: _r
        )
        assert speculative.verify_scratch_in_pool() is raw_gate
        # Without speculation the kernel answer is never consulted.
        assert kda_verify_scratch_in_pool(plain.server_args, plain.attn_config) is False

    # A PD prefill role plans no verify workspace, so its pool stages none.
    monkeypatch.setattr(
        kda_ops, "kda_batched_replay_uses_raw_gate", lambda *a, **k: True
    )
    speculative.server_args.disaggregation_mode = "prefill"
    assert speculative.workspace_bytes() == 0
    assert speculative.verify_scratch_in_pool() is False

    def arena_bytes(max_bs: int) -> int:
        recipe = _kimi_k3_recipe(
            max_bs=max_bs, speculative_algorithm="eagle", speculative_num_draft_tokens=2
        )
        recipe.cache_budget_bytes = 0
        recipe.probe_batch_rows = 2
        assert recipe.verify_scratch_in_pool() is True
        return recipe.setup().spec.memory_plan.arena_bytes

    monkeypatch.setattr(
        kda_ops, "kda_batched_replay_uses_raw_gate", lambda *a, **k: True
    )
    # Its verify scratch is a row per request of the bound pool, probe or not.
    assert arena_bytes(8) < arena_bytes(64)


def test_each_refusal_turns_the_probe_off_and_names_itself() -> None:
    def refusal(disable=False, eager=False):
        args = SimpleNamespace(
            disable_cudagraph_memory_reserve=disable, enforce_eager=eager
        )
        return device._cudagraph_probe_refusal(args)

    assert refusal() is None
    assert "--disable-cudagraph-memory-reserve" in refusal(disable=True)
    assert "--enforce-eager" in refusal(eager=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
