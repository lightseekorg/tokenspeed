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

"""The probe arena the reserve binds first: every family has to be able to hold it.

The reserve is on by default, so the probe arena floor is a boot-path
constant for every cache family, not only the flat paged ones. A family whose
per-group demand sizes its own parents (Kimi-K3, GLM-5.3-flash, DeepSeek V4)
rejects a parent count that cannot admit a single token, and that rejection is
raised inside ``create_attn_components`` before a model executor exists.
"""

from __future__ import annotations

import ast
import pathlib
import sys
from types import SimpleNamespace

import pytest

# CI runs this file as a script; a developer runs pytest from the repo root.
# Both entry points need the test dir (siblings) and the root (test.runtime.*).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.execution.cudagraph_memory import (  # noqa: E402
    probe_arena_parent_blocks,
)

# The kimi recipe fixture below builds its arena for this batch.
_MAX_BS = 8
_MAX_FORWARD_TOKENS = 8192
_CONTEXT_LEN = 4096
_PROBE_FLOOR = probe_arena_parent_blocks(
    max_forward_tokens=_MAX_FORWARD_TOKENS, context_len=_CONTEXT_LEN
)
# What create_attn_components hands a probe: an arena sized by block count
# needs no memory profile, so there is no budget to size or cap it with.
_PROBE_BUDGET = 0


def _kimi_k3_recipe():
    from test.runtime.conftest import kimi_recipe

    return kimi_recipe(max_bs=8)


def _glm53_flash_recipe():
    from test.runtime.test_glm53_flash_cache_spec import _recipe

    import torch

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


_RECIPES = {
    "kimi_k3": _kimi_k3_recipe,
    "glm53_flash": _glm53_flash_recipe,
    "deepseek_v4": _deepseek_v4_recipe,
}


@pytest.mark.parametrize("family", sorted(_RECIPES))
def test_the_probe_arena_is_sizeable_for_every_family_it_boots(family: str) -> None:
    """The default boot path binds this arena; a family that cannot is a crash.

    ``--disable-cudagraph-memory-reserve`` is off by default, so every family
    runs the probe. A recipe that sizes parents from per-group demand needs
    enough parents for the state groups' rolling checkpoints before it can
    admit a single token, so a flat constant cannot serve all of them.
    """
    recipe = _RECIPES[family]()
    recipe.cache_budget_bytes = _PROBE_BUDGET
    recipe.num_lcm_blocks_override = _PROBE_FLOOR

    setup = recipe.setup()

    assert setup.spec.token_capacity > 0
    assert setup.spec.memory_plan.num_lcm_blocks >= _PROBE_FLOOR


def test_the_probe_block_floor_covers_every_extend_the_probe_fabricates() -> None:
    """The floor is the widest fabricated extend, not the tuning batch.

    The bucket ladder is clamped by the per-forward token budget and nothing
    else, so a floor derived from the tuning batch (which is additionally
    clamped by the concurrency) is short whenever the server runs fewer
    requests than the budget spans -- and a short floor is a capture that
    indexes a page id the block tables reject. The recipe's own per-group
    demand is capped the same way, see the next test.
    """
    from tokenspeed.runtime.execution.prefill_graph import (
        dummy_batch_size,
        get_prefill_token_buckets,
    )

    for budget, context_len in ((8192, 4096), (16384, 4096), (32768, 8192)):
        config = SimpleNamespace(
            disable_prefill_graph=False,
            prefill_graph_max_tokens=budget,
            prefill_graph_capture_sizes=None,
            chunked_prefill_size=budget,
        )
        floor = probe_arena_parent_blocks(
            max_forward_tokens=budget, context_len=context_len
        )
        widest = max(get_prefill_token_buckets(config))

        assert floor >= dummy_batch_size(widest, context_len), (budget, context_len)
        assert floor == -(-budget // context_len)

    # Every case above is an exact multiple, where a floored floor reads the
    # same. A ragged split needs one more row than a division gives.
    for budget, context_len in ((8192, 3000), (8193, 4096), (5000, 4096), (100, 64)):
        floor = probe_arena_parent_blocks(
            max_forward_tokens=budget, context_len=context_len
        )
        assert floor == dummy_batch_size(budget, context_len), (budget, context_len)
        assert floor * context_len >= budget, (budget, context_len)
    assert probe_arena_parent_blocks(max_forward_tokens=8192, context_len=3000) == 3


def test_the_override_reaches_the_plan_through_the_factory_seam() -> None:
    """The override has to survive the hop the boot path actually takes.

    Every other test sets it on a recipe that already exists; production
    passes it through ``prepare_cache_setup``, and a hop that drops it binds
    a budget-sized arena before the memory profile has run.
    """
    from test.runtime.conftest import kimi_recipe

    from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import (
        prepare_cache_setup,
    )

    template = kimi_recipe(max_bs=_MAX_BS)
    setup = prepare_cache_setup(
        family="kimi_k3",
        server_args=template.server_args,
        model_config=template.model_config,
        attn_config=template.attn_config,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=_PROBE_BUDGET,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
        num_lcm_blocks_override=64,
    )

    assert setup.spec.memory_plan.num_lcm_blocks == 64


def test_the_factory_forwards_the_override_and_the_reserve_it_was_given() -> None:
    """The two numbers the boot step produces only matter if they arrive.

    A probe build that reaches the memory profile is not a probe, and a
    rebuild that profiles without the projection is not a reserve. Both read
    the same way from every test that stops at ``build_components``, and both
    are one hard-coded keyword away, so the forwarding is asserted here.
    """
    import ast
    import pathlib as _pathlib

    def _forwarded(module: str, function: str, callee: str, keyword: str) -> str:
        path = (
            _pathlib.Path(__file__).resolve().parents[2]
            / "python"
            / "tokenspeed"
            / "runtime"
            / "layers"
            / "attention"
            / module
        )
        body = next(
            node
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.FunctionDef) and node.name == function
        )
        call = next(
            node
            for node in ast.walk(body)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", getattr(node.func, "attr", None)) == callee
        )
        value = next(kw.value for kw in call.keywords if kw.arg == keyword)
        return ast.unparse(value)

    assert (
        _forwarded(
            "registry.py",
            "create_attn_components",
            "profile_available_cache_memory_bytes",
            "graph_reserve_bytes",
        )
        == "graph_reserve_bytes"
    )
    assert (
        _forwarded(
            "registry.py",
            "create_attn_components",
            "prepare_cache_setup",
            "num_lcm_blocks_override",
        )
        == "num_lcm_blocks_override"
    )
    assert (
        _forwarded(
            "kv_cache/recipes/setup.py",
            "prepare_cache_setup",
            "cache_recipe",
            "num_lcm_blocks_override",
        )
        == "num_lcm_blocks_override"
    )


def test_the_probe_arena_does_not_grow_with_max_num_seqs() -> None:
    """The block-count floor is not the only thing that sizes a probe arena.

    Every recipe's ``parents_needed`` is per-request demand at the configured
    concurrency, and it is the arm the ``max()`` usually takes -- so without
    capping the scheduler limits too, a boot at ``--max-num-seqs 512`` binds a
    multi-GiB arena for four captures and then drops it, and the profile that
    sizes the real cache counts every byte of it as spent.
    """
    from test.runtime.conftest import kimi_recipe

    def arena_bytes(max_bs: int) -> int:
        recipe = kimi_recipe(max_bs=max_bs)
        recipe.cache_budget_bytes = _PROBE_BUDGET
        recipe.num_lcm_blocks_override = _PROBE_FLOOR
        return recipe.setup().spec.memory_plan.arena_bytes

    assert arena_bytes(8) == arena_bytes(512)
    # A cap, not a floor: a server smaller than the probe's batch stays smaller.
    assert arena_bytes(1) < arena_bytes(512)


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


def test_the_probe_build_is_the_one_that_takes_the_override() -> None:
    """The ternary direction and its argument are the whole probe arena.

    Inverted, the probe build profiles memory and binds a served-size arena
    before the rebuild binds a second one. Fed a constant, the floor stops
    tracking the per-forward token budget the fabricated extends are sized by.
    """
    build = _build_device_side()
    call = next(
        node
        for node in ast.walk(build)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "build_components"
        and any(kw.arg == "num_lcm_blocks_override" for kw in node.keywords)
    )
    override = next(
        kw.value for kw in call.keywords if kw.arg == "num_lcm_blocks_override"
    )

    assert isinstance(override, ast.IfExp)
    assert isinstance(override.test, ast.Name) and override.test.id == "probing"
    assert isinstance(override.orelse, ast.Constant) and override.orelse.value is None
    assert getattr(override.body.func, "id", None) == "probe_arena_parent_blocks"
    assert {kw.arg: ast.unparse(kw.value) for kw in override.body.keywords} == {
        "max_forward_tokens": "max_forward_tokens",
        "context_len": "model_config.context_len",
    }


def test_only_a_pool_staged_verify_scratch_refuses_the_probe() -> None:
    """Speculation alone does not refuse a probe; aliasing the pool does.

    Raw-gate KDA replay reuses the committed conv slab as verify scratch and
    asks for a row per request at the serving concurrency -- against
    whichever pool is bound. Sizing a probe arena for that was measured at
    31 GiB on Kimi-K3 at ``--max-num-seqs 256``, so that family, and only
    that family, boots without one. Every other speculative boot probes, so
    ``--gpu-memory-utilization`` means the same thing everywhere.
    """
    from test.runtime.conftest import kimi_recipe

    from tokenspeed.runtime.layers.attention.kv_cache.recipes.base import (
        kda_verify_scratch_in_pool,
    )

    for family in sorted(_RECIPES):
        assert _RECIPES[family]().verify_scratch_in_pool() is False, family

    plain = kimi_recipe(max_bs=8)
    speculative = kimi_recipe(
        max_bs=8, speculative_algorithm="eagle", speculative_num_draft_tokens=2
    )
    assert plain.verify_scratch_in_pool() is False
    # Decided by the kernel registry for this dtype and geometry, not by us.
    assert speculative.verify_scratch_in_pool() is kda_verify_scratch_in_pool(
        speculative.server_args, speculative.attn_config
    )


@pytest.mark.parametrize("family", sorted(_RECIPES))
def test_the_probe_budget_covers_the_arena_it_binds(family: str) -> None:
    """The reported budget is what the probe arena actually costs.

    ``CacheSetup.cache_budget_bytes`` is the number the storage report and the
    memory summary attribute to the cache, so under an override it has to
    cover the bound plan's bytes plus the fixed workspace -- an off-by-one
    parent here under-reports a real allocation.
    """
    recipe = _RECIPES[family]()
    recipe.cache_budget_bytes = _PROBE_BUDGET
    # Large enough that every family's per-group demand is satisfiable.
    recipe.num_lcm_blocks_override = 1024

    setup = recipe.setup()

    assert setup.cache_budget_bytes >= (
        setup.spec.memory_plan.arena_bytes + setup.fixed_workspace_bytes
    )


def test_the_probe_override_wins_over_the_budget() -> None:
    """The override replaces the budget search, it does not merely cap it."""
    recipe = _kimi_k3_recipe()
    recipe.cache_budget_bytes = _PROBE_BUDGET
    recipe.num_lcm_blocks_override = 1024

    setup = recipe.setup()

    assert setup.spec.memory_plan.num_lcm_blocks == 1024


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
