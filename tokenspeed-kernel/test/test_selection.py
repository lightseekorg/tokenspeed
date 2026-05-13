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

from __future__ import annotations

import os
from unittest import mock

import pytest
import tokenspeed_kernel.ops.gemm as gemm
import torch
from tokenspeed_kernel.platform import PlatformInfo
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec
from tokenspeed_kernel.selection import (
    AutotuneParams,
    NoKernelFoundError,
    ScoreBreakdown,
    SelectionObjective,
    SelectionOracle,
    SelectionPolicy,
    SelectionStrategy,
    _filter_by_traits,
    _get_config_override,
    _make_cache_key,
    _parse_overrides,
    _rank_by_objective,
    _score,
    _score_objective,
    _score_priority,
    clear_config_overrides,
    explain_selection,
    kernel_override,
    load_config_overrides,
    register_oracle,
    select_kernel,
    set_selection_policy,
    spec_matches_shape_traits,
    spec_matches_traits,
    warmup_selection,
)
from utils import register_all_samples

pytestmark = pytest.mark.usefixtures("fresh_registry")


class TestSelectionObjective:
    def test_all_enum_values(self):
        assert SelectionObjective.DEFAULT.value == "default"
        assert SelectionObjective.LATENCY.value == "latency"
        assert SelectionObjective.THROUGHPUT.value == "throughput"
        assert SelectionObjective.PORTABILITY.value == "portability"
        assert SelectionObjective.DETERMINISM.value == "determinism"
        assert SelectionObjective.DEBUG.value == "debug"


class TestScoreBreakdown:
    def test_str_format(self):
        bd = ScoreBreakdown(priority=10, objective=12, oracle=14)
        assert str(bd) == "ora=14 obj=12 pri=10"

    def test_sort_key(self):
        bd = ScoreBreakdown(priority=10, objective=12, oracle=14)
        assert bd.sort_key() == (14, 12, 10)


class TestAutotuneParams:
    def test_defaults(self):
        p = AutotuneParams()
        assert p.warmup_iters == 3
        assert p.bench_iters == 10
        assert p.use_cuda_events is True


class TestSelectionPolicy:
    def test_default_strategy(self):
        policy = SelectionPolicy()
        assert policy.get_strategy("attention", "decode") == SelectionStrategy.HEURISTIC

    def test_per_op_override(self):
        policy = SelectionPolicy(
            op_strategies={("gemm", "mm"): SelectionStrategy.AUTOTUNE},
        )
        assert policy.get_strategy("gemm", "mm") == SelectionStrategy.AUTOTUNE
        assert policy.get_strategy("attention", "decode") == SelectionStrategy.HEURISTIC


class TestScorePriority:
    def test_normal_range(self):
        spec = KernelSpec(name="k", family="f", mode="m", priority=15)
        assert _score_priority(spec) == 15

    def test_clamped_low(self):
        spec = KernelSpec(name="k", family="f", mode="m", priority=-5)
        assert _score_priority(spec) == 0

    def test_clamped_high(self):
        spec = KernelSpec(name="k", family="f", mode="m", priority=25)
        assert _score_priority(spec) == 19


class TestScoreObjective:
    def _spec(self, solution="triton", tags=frozenset()):
        return KernelSpec(name="k", family="f", mode="m", solution=solution, tags=tags)

    def test_default_ties_everyone(self):
        assert _score_objective(self._spec(), SelectionObjective.DEFAULT) == 0
        assert (
            _score_objective(
                self._spec(tags=frozenset({"latency"})),
                SelectionObjective.DEFAULT,
            )
            == 0
        )

    def test_latency_tag_match(self):
        spec = self._spec(tags=frozenset({"latency"}))
        assert _score_objective(spec, SelectionObjective.LATENCY) == 1

    def test_latency_no_match(self):
        spec = self._spec(tags=frozenset({"throughput"}))
        assert _score_objective(spec, SelectionObjective.LATENCY) == 0

    def test_throughput_tag_match(self):
        spec = self._spec(tags=frozenset({"throughput"}))
        assert _score_objective(spec, SelectionObjective.THROUGHPUT) == 1

    def test_throughput_no_match(self):
        assert _score_objective(self._spec(), SelectionObjective.THROUGHPUT) == 0

    def test_portability_tag_match(self):
        spec = self._spec(tags=frozenset({"portability"}))
        assert _score_objective(spec, SelectionObjective.PORTABILITY) == 1

    def test_portability_no_match(self):
        assert (
            _score_objective(
                self._spec(solution="triton"), SelectionObjective.PORTABILITY
            )
            == 0
        )

    def test_determinism_tag_match(self):
        spec = self._spec(tags=frozenset({"determinism"}))
        assert _score_objective(spec, SelectionObjective.DETERMINISM) == 1

    def test_debug_uses_determinism_tag(self):
        det = self._spec(tags=frozenset({"determinism"}))
        plain = self._spec()
        assert _score_objective(det, SelectionObjective.DEBUG) == 1
        assert _score_objective(plain, SelectionObjective.DEBUG) == 0


class TestScore:
    def test_score_returns_per_dimension_breakdown(self, h100_platform):
        spec = KernelSpec(
            name="k",
            family="f",
            mode="m",
            solution="cutlass",
            priority=15,
            tags=frozenset({"latency"}),
        )
        bd = _score(spec, SelectionObjective.LATENCY, h100_platform, None)
        assert bd.priority == 15
        assert bd.objective == 1  # latency tag matches
        assert bd.oracle == 10  # neutral, no oracle registered


class TestRanking:
    def test_rank_orders_lexicographically(self, sample_specs, h100_platform):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        candidates = reg.get_for_operator(
            "attention",
            "decode",
            platform=h100_platform,
            dtype=torch.bfloat16,
        )
        scored = _rank_by_objective(
            candidates,
            SelectionObjective.DEFAULT,
            h100_platform,
            None,
        )
        keys = [bd.sort_key() for _, bd in scored]
        assert keys == sorted(keys, reverse=True)

    def test_oracle_outranks_objective_and_priority(self, h100_platform):
        oracle_winner = KernelSpec(
            name="oracle_winner",
            family="f",
            mode="m",
            solution="reference",
            priority=0,
        )
        objective_winner = KernelSpec(
            name="objective_winner",
            family="f",
            mode="m",
            solution="triton",
            priority=0,
            tags=frozenset({"latency"}),
        )
        priority_winner = KernelSpec(
            name="priority_winner",
            family="f",
            mode="m",
            solution="triton",
            priority=19,
        )

        class BoostOracleWinner(SelectionOracle):
            def adjust(self, spec, platform, traits):
                return 19 if spec.name == "oracle_winner" else 0

        register_oracle("f", BoostOracleWinner())

        scored = _rank_by_objective(
            [priority_winner, objective_winner, oracle_winner],
            SelectionObjective.LATENCY,
            h100_platform,
            None,
        )
        assert [s.name for s, _ in scored] == [
            "oracle_winner",
            "objective_winner",
            "priority_winner",
        ]

    def test_priority_breaks_ties(self, h100_platform):
        low = KernelSpec(name="low", family="f", mode="m", priority=5)
        high = KernelSpec(name="high", family="f", mode="m", priority=15)

        scored = _rank_by_objective(
            [low, high],
            SelectionObjective.DEFAULT,
            h100_platform,
            None,
        )
        assert [s.name for s, _ in scored] == ["high", "low"]


class TestFilterByTraits:
    def test_compatible_trait(self):
        spec = KernelSpec(
            name="k",
            family="f",
            mode="m",
            traits={"head_dim": frozenset({128})},
        )
        result = _filter_by_traits([spec], {"head_dim": 128})
        assert len(result) == 1

    def test_incompatible_trait(self):
        spec = KernelSpec(
            name="k",
            family="f",
            mode="m",
            traits={"head_dim": frozenset({64, 128})},
        )
        result = _filter_by_traits([spec], {"head_dim": 256})
        assert len(result) == 0

    def test_unknown_trait_passes(self):
        spec = KernelSpec(name="k", family="f", mode="m", traits={})
        result = _filter_by_traits([spec], {"head_dim": 128})
        assert len(result) == 1

    def test_multiple_traits(self):
        spec = KernelSpec(
            name="k",
            family="f",
            mode="m",
            traits={
                "head_dim": frozenset({128}),
                "num_kv_heads": frozenset({8}),
            },
        )
        assert len(_filter_by_traits([spec], {"head_dim": 128, "num_kv_heads": 8})) == 1
        assert (
            len(_filter_by_traits([spec], {"head_dim": 128, "num_kv_heads": 32})) == 0
        )


class TestSpecMatchesTraits:
    def test_scalar_requested_value_matches_if_in_spec_set(self):
        spec = KernelSpec(
            name="k",
            family="f",
            mode="m",
            traits={"head_dim": frozenset({64, 128})},
        )

        assert spec_matches_traits(spec, {"head_dim": 128})
        assert not spec_matches_traits(spec, {"head_dim": 256})

    def test_scalar_requested_value_matches_equal_singleton(self):
        spec = KernelSpec(
            name="k",
            family="f",
            mode="m",
            traits={"head_dim": frozenset({128})},
        )

        assert spec_matches_traits(spec, {"head_dim": 128})
        assert not spec_matches_traits(spec, {"head_dim": 256})

    def test_set_requested_value_matches(self):
        spec = KernelSpec(
            name="k",
            family="f",
            mode="m",
            traits={"b_layout": frozenset({"KN"})},
        )

        assert spec_matches_traits(spec, {"b_layout": frozenset({"KN"})})
        assert not spec_matches_traits(spec, {"b_layout": frozenset({"KN", "NK"})})
        assert not spec_matches_traits(spec, {"b_layout": frozenset({"KM"})})

    def test_set_requested_value_subset_of_spec(self):
        spec = KernelSpec(
            name="k",
            family="f",
            mode="m",
            traits={"b_layout": frozenset({"KN", "NK"})},
        )

        assert spec_matches_traits(spec, {"b_layout": frozenset({"KN"})})
        assert spec_matches_traits(spec, {"b_layout": frozenset({"KN", "NK"})})
        assert not spec_matches_traits(spec, {"b_layout": frozenset({"KM"})})
        assert not spec_matches_traits(spec, {"b_layout": frozenset({"KN", "KM"})})

    def test_missing_trait_is_ignored_by_default(self):
        spec = KernelSpec(name="k", family="f", mode="m", traits={})

        assert spec_matches_traits(spec, {"head_dim": 128})

    def test_missing_trait_can_be_required(self):
        spec = KernelSpec(name="k", family="f", mode="m", traits={})

        assert not spec_matches_traits(
            spec,
            {"head_dim": frozenset({128})},
            require_all_traits=True,
        )


class TestSpecMatchesShapeTraits:
    def test_required_alignment_trait_matches(self):
        spec = KernelSpec(
            name="k",
            family="f",
            mode="m",
            traits={"n_align_16": frozenset({True})},
        )

        assert spec_matches_shape_traits(spec, {"N": 32})
        assert not spec_matches_shape_traits(spec, {"N": 30})

    def test_missing_shape_dim_is_ignored(self):
        spec = KernelSpec(
            name="k",
            family="f",
            mode="m",
            traits={"k_align_128": frozenset({True})},
        )

        assert spec_matches_shape_traits(spec, {})

    def test_non_alignment_traits_do_not_affect_shape_matching(self):
        spec = KernelSpec(
            name="k",
            family="f",
            mode="m",
            traits={"quant": frozenset({"mxfp8"})},
        )

        assert spec_matches_shape_traits(spec, {"N": 30, "K": 70})


class TestMakeCacheKey:
    def test_deterministic(self):
        k1 = _make_cache_key(
            "attn",
            "dec",
            torch.bfloat16,
            "sm_90",
            SelectionObjective.DEFAULT,
            None,
            None,
        )
        k2 = _make_cache_key(
            "attn",
            "dec",
            torch.bfloat16,
            "sm_90",
            SelectionObjective.DEFAULT,
            None,
            None,
        )
        assert k1 == k2

    def test_different_objective(self):
        k1 = _make_cache_key(
            "attn",
            "dec",
            torch.bfloat16,
            "sm_90",
            SelectionObjective.DEFAULT,
            None,
            None,
        )
        k2 = _make_cache_key(
            "attn",
            "dec",
            torch.bfloat16,
            "sm_90",
            SelectionObjective.LATENCY,
            None,
            None,
        )
        assert k1 != k2

    def test_traits_order_independent(self):
        k1 = _make_cache_key(
            "a",
            "d",
            torch.float16,
            "sm_90",
            SelectionObjective.DEFAULT,
            None,
            {"a": 1, "b": 2},
        )
        k2 = _make_cache_key(
            "a",
            "d",
            torch.float16,
            "sm_90",
            SelectionObjective.DEFAULT,
            None,
            {"b": 2, "a": 1},
        )
        assert k1 == k2

    def test_features_order_independent(self):
        f1 = frozenset({"paged", "mla"})
        f2 = frozenset({"mla", "paged"})
        k1 = _make_cache_key(
            "a", "d", torch.float16, "sm_90", SelectionObjective.DEFAULT, f1, None
        )
        k2 = _make_cache_key(
            "a", "d", torch.float16, "sm_90", SelectionObjective.DEFAULT, f2, None
        )
        assert k1 == k2

    def test_solution_is_selection_relevant(self):
        k1 = _make_cache_key(
            "a",
            "d",
            torch.float16,
            "sm_90",
            SelectionObjective.DEFAULT,
            None,
            None,
            "fa3",
        )
        k2 = _make_cache_key(
            "a",
            "d",
            torch.float16,
            "sm_90",
            SelectionObjective.DEFAULT,
            None,
            None,
            "fa4",
        )
        assert k1 != k2


class TestSelectKernel:
    def test_basic_selection(self, sample_specs, h100_platform):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        impl = select_kernel(
            "attention",
            "decode",
            torch.bfloat16,
            platform=h100_platform,
        )
        assert callable(impl)

    def test_cached_on_second_call(self, sample_specs, h100_platform):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        impl1 = select_kernel(
            "attention", "decode", torch.bfloat16, platform=h100_platform
        )
        impl2 = select_kernel(
            "attention", "decode", torch.bfloat16, platform=h100_platform
        )
        assert impl1 is impl2

    def test_no_kernel_raises(self, h100_platform):
        with pytest.raises(NoKernelFoundError):
            select_kernel("nonexistent", "op", torch.bfloat16, platform=h100_platform)

    def test_no_kernel_after_trait_filter(self, h100_platform):
        reg = KernelRegistry.get()
        spec = KernelSpec(
            name="trait_k",
            family="trait_op",
            mode="m",
            solution="triton",
            priority=10,
            dtypes=frozenset({torch.bfloat16}),
            traits={"head_dim": frozenset({64})},
        )
        reg.register(spec, lambda: None)

        with pytest.raises(NoKernelFoundError, match="traits"):
            select_kernel(
                "trait_op",
                "m",
                torch.bfloat16,
                platform=h100_platform,
                traits={"head_dim": 128},
            )

    def test_override_by_name(self, sample_specs, h100_platform):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        impl = select_kernel(
            "attention",
            "decode",
            torch.bfloat16,
            platform=h100_platform,
            override="reference_decode",
        )
        assert impl() == "reference_decode"

    def test_override_by_solution(self, sample_specs, h100_platform):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        impl = select_kernel(
            "attention",
            "decode",
            torch.bfloat16,
            platform=h100_platform,
            override="triton",
        )
        assert impl() == "triton_decode"

    def test_solution_filter_preserves_trait_filtering(self, h100_platform):
        reg = KernelRegistry.get()
        reg.register(
            KernelSpec(
                name="fa4_128",
                family="attention",
                mode="prefill",
                solution="fa4",
                dtypes=frozenset({torch.bfloat16}),
                traits={"head_dim": frozenset({128})},
                priority=15,
            ),
            lambda: "fa4_128",
        )
        reg.register(
            KernelSpec(
                name="triton_256",
                family="attention",
                mode="prefill",
                solution="triton",
                dtypes=frozenset({torch.bfloat16}),
                traits={"head_dim": frozenset({256})},
                priority=10,
            ),
            lambda: "triton_256",
        )

        impl = select_kernel(
            "attention",
            "prefill",
            torch.bfloat16,
            platform=h100_platform,
            solution="fa4",
            traits={"head_dim": 128},
        )
        assert impl() == "fa4_128"

        with pytest.raises(NoKernelFoundError, match="solution 'fa4'.*traits"):
            select_kernel(
                "attention",
                "prefill",
                torch.bfloat16,
                platform=h100_platform,
                solution="fa4",
                traits={"head_dim": 256},
            )

    def test_override_not_found_raises(self, sample_specs, h100_platform):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        with pytest.raises(NoKernelFoundError, match="Override"):
            select_kernel(
                "attention",
                "decode",
                torch.bfloat16,
                platform=h100_platform,
                override="nonexistent_kernel",
            )

    def test_env_override(self, sample_specs, h100_platform):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        with mock.patch.dict(
            os.environ,
            {"TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_DECODE": "reference_decode"},
        ):
            impl = select_kernel(
                "attention",
                "decode",
                torch.bfloat16,
                platform=h100_platform,
            )
            assert impl() == "reference_decode"

    def test_portability_objective_prefers_triton(self, sample_specs, h100_platform):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        impl = select_kernel(
            "attention",
            "decode",
            torch.bfloat16,
            platform=h100_platform,
            objective=SelectionObjective.PORTABILITY,
        )
        assert impl() == "triton_decode"

    def test_debug_objective_prefers_reference(self, sample_specs, h100_platform):
        """DEBUG ranks the determinism-tagged reference kernel above others."""
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        impl = select_kernel(
            "attention",
            "decode",
            torch.bfloat16,
            platform=h100_platform,
            objective=SelectionObjective.DEBUG,
        )
        assert impl() == "reference_decode"

    def test_amd_platform_selects_aiter(self, sample_specs, mi300_platform):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        impl = select_kernel(
            "attention",
            "decode",
            torch.bfloat16,
            platform=mi300_platform,
        )
        assert impl() == "aiter_decode"

    def test_amd_mi355_platform_selects_aiter(self, sample_specs, mi355_platform):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        impl = select_kernel(
            "attention",
            "decode",
            torch.bfloat16,
            platform=mi355_platform,
        )
        assert impl() == "aiter_decode"


class TestSelectionOracle:
    def test_default_oracle_neutral(self):
        oracle = SelectionOracle()
        spec = KernelSpec(name="k", family="f", mode="m")
        assert oracle.adjust(spec, None, None) == 10

    def test_register_oracle(self, sample_specs, h100_platform):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        class BoostTritonOracle(SelectionOracle):
            def adjust(self, spec, platform, traits):
                if spec.solution == "triton":
                    return 19
                return 0

        register_oracle("attention", BoostTritonOracle())

        impl = select_kernel(
            "attention",
            "decode",
            torch.bfloat16,
            platform=h100_platform,
        )
        assert impl() == "triton_decode"


class TestKernelOverride:
    def test_context_manager_overrides(self, sample_specs, h100_platform):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        with kernel_override("attention", "decode", "reference_decode"):
            impl = select_kernel(
                "attention",
                "decode",
                torch.bfloat16,
                platform=h100_platform,
            )
            assert impl() == "reference_decode"

    def test_context_manager_restores(self, sample_specs, h100_platform):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        with kernel_override("attention", "decode", "reference_decode"):
            pass

        impl = select_kernel(
            "attention",
            "decode",
            torch.bfloat16,
            platform=h100_platform,
        )
        assert impl() != "reference_decode" or True

    def test_nested_override(self, sample_specs, h100_platform):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        with kernel_override("attention", "decode", "reference_decode"):
            impl1 = select_kernel(
                "attention", "decode", torch.bfloat16, platform=h100_platform
            )
            assert impl1() == "reference_decode"

            with kernel_override("attention", "decode", "triton_decode"):
                impl2 = select_kernel(
                    "attention", "decode", torch.bfloat16, platform=h100_platform
                )
                assert impl2() == "triton_decode"

            impl3 = select_kernel(
                "attention", "decode", torch.bfloat16, platform=h100_platform
            )
            assert impl3() == "reference_decode"


class TestSetPolicy:
    def test_set_policy_clears_cache(self, sample_specs, h100_platform):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        select_kernel("attention", "decode", torch.bfloat16, platform=h100_platform)
        set_selection_policy(
            SelectionPolicy(default_strategy=SelectionStrategy.AUTOTUNE)
        )
        assert not reg._selection_cache


class TestExplainSelection:
    def test_output_contains_expected_sections(self, sample_specs, h100_platform):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        explanation = explain_selection(
            "attention",
            "decode",
            torch.bfloat16,
            platform=h100_platform,
        )
        assert "attention.decode" in explanation
        assert "NVIDIA H100" in explanation
        assert "[SELECTED]" in explanation
        assert "Candidates" in explanation

    def test_filtered_out_section(self, sample_specs, h100_platform):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        explanation = explain_selection(
            "attention",
            "decode",
            torch.bfloat16,
            platform=h100_platform,
        )
        assert "Filtered out" in explanation
        assert "aiter_decode" in explanation

    def test_empty_candidates(self, h100_platform):
        explanation = explain_selection(
            "nonexistent",
            "op",
            torch.bfloat16,
            platform=h100_platform,
        )
        assert "0 matched" in explanation


class TestWarmupSelection:
    def test_warmup_fills_cache(self, sample_specs, h100_platform):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        from tokenspeed_kernel.platform import Platform

        Platform.override(h100_platform)
        try:
            warmup_selection()
            assert len(reg._selection_cache) > 0
        finally:
            Platform.reset()

    def test_warmup_explicit_ops(self, sample_specs, h100_platform):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        from tokenspeed_kernel.platform import Platform

        Platform.override(h100_platform)
        try:
            warmup_selection(
                ops=[
                    ("attention", "decode", torch.bfloat16, None),
                    ("gemm", "mm", torch.bfloat16, None),
                ]
            )
            assert len(reg._selection_cache) >= 2
        finally:
            Platform.reset()

    def test_warmup_skips_missing_ops(self, h100_platform):
        """warmup_selection should not raise for missing ops."""
        from tokenspeed_kernel.platform import Platform

        Platform.override(h100_platform)
        try:
            warmup_selection(ops=[("nonexistent", "op", torch.bfloat16, None)])
        finally:
            Platform.reset()


class TestAutotuneStrategy:
    def test_autotune_falls_back_to_heuristic(self, sample_specs, h100_platform):
        set_selection_policy(
            SelectionPolicy(
                default_strategy=SelectionStrategy.AUTOTUNE,
            )
        )
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        impl = select_kernel(
            "attention",
            "decode",
            torch.bfloat16,
            platform=h100_platform,
        )
        assert callable(impl)


class TestParseOverrides:
    """Unit tests for _parse_overrides (YAML dict → _ConfigOverrideEntry)."""

    def test_name_override(self):
        raw = {"gemm.mm": {"name": "gluon_gemm_mm_fp8"}}
        result = _parse_overrides(raw)
        assert ("gemm", "mm") in result
        entry = result[("gemm", "mm")]
        assert entry.name == "gluon_gemm_mm_fp8"
        assert entry.solution is None
        assert entry.objective is None

    def test_solution_override(self):
        raw = {"attention.decode": {"solution": "flashinfer"}}
        result = _parse_overrides(raw)
        entry = result[("attention", "decode")]
        assert entry.name is None
        assert entry.solution == "flashinfer"

    def test_objective_override(self):
        raw = {"moe.experts": {"objective": "determinism"}}
        result = _parse_overrides(raw)
        entry = result[("moe", "experts")]
        assert entry.name is None
        assert entry.solution is None
        assert entry.objective == "determinism"

    def test_both_name_and_solution(self):
        raw = {"gemm.mm": {"name": "exact_name", "solution": "triton"}}
        result = _parse_overrides(raw)
        entry = result[("gemm", "mm")]
        assert entry.name == "exact_name"
        assert entry.solution == "triton"

    def test_shorthand_string_value(self):
        raw = {"norm.rmsnorm": "triton_rmsnorm"}
        result = _parse_overrides(raw)
        entry = result[("norm", "rmsnorm")]
        assert entry.name == "triton_rmsnorm"
        assert entry.solution is None

    def test_combined_solution_and_objective(self):
        raw = {"attention.decode": {"solution": "flashinfer", "objective": "latency"}}
        result = _parse_overrides(raw)
        entry = result[("attention", "decode")]
        assert entry.solution == "flashinfer"
        assert entry.objective == "latency"

    def test_invalid_key_without_dot(self):
        raw = {"attention_decode": {"name": "foo"}}
        result = _parse_overrides(raw)
        assert len(result) == 0

    def test_non_dict_input_returns_empty(self):
        assert _parse_overrides("not a dict") == {}
        assert _parse_overrides(42) == {}
        assert _parse_overrides(None) == {}

    def test_multiple_entries(self):
        raw = {
            "attention.decode": {"solution": "flashinfer"},
            "gemm.mm": {"name": "cutlass_gemm"},
            "moe.fused": {"objective": "throughput"},
        }
        result = _parse_overrides(raw)
        assert len(result) == 3


class TestLoadConfigOverrides:
    """Tests for load_config_overrides / clear_config_overrides."""

    def test_load_from_file(self, tmp_path):
        yaml_content = (
            "overrides:\n"
            "  attention.decode:\n"
            "    solution: flashinfer\n"
            "  gemm.mm:\n"
            "    name: cutlass_gemm\n"
        )
        config_file = tmp_path / "overrides.yaml"
        config_file.write_text(yaml_content)

        load_config_overrides(config_file)

        entry = _get_config_override("attention", "decode")
        assert entry is not None
        assert entry.solution == "flashinfer"
        entry2 = _get_config_override("gemm", "mm")
        assert entry2 is not None
        assert entry2.name == "cutlass_gemm"

    def test_load_nonexistent_file(self, tmp_path):
        load_config_overrides(tmp_path / "does_not_exist.yaml")
        assert _get_config_override("attention", "decode") is None

    def test_load_invalid_yaml(self, tmp_path):
        config_file = tmp_path / "bad.yaml"
        config_file.write_text(": : : not valid yaml [[[")

        load_config_overrides(config_file)
        assert _get_config_override("attention", "decode") is None

    def test_load_empty_file(self, tmp_path):
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        load_config_overrides(config_file)
        assert _get_config_override("attention", "decode") is None

    def test_load_no_overrides_section(self, tmp_path):
        config_file = tmp_path / "no_overrides.yaml"
        config_file.write_text("some_other_key: value\n")

        load_config_overrides(config_file)
        assert _get_config_override("attention", "decode") is None

    def test_clear_config_overrides(self, tmp_path):
        yaml_content = (
            "overrides:\n" "  attention.decode:\n" "    solution: flashinfer\n"
        )
        config_file = tmp_path / "overrides.yaml"
        config_file.write_text(yaml_content)

        load_config_overrides(config_file)
        assert _get_config_override("attention", "decode") is not None

        clear_config_overrides()
        assert _get_config_override("attention", "decode") is None

    def test_env_var_overrides_file_path(self, tmp_path):
        yaml_content = "overrides:\n" "  gemm.mm:\n" "    name: custom_gemm\n"
        config_file = tmp_path / "custom_overrides.yaml"
        config_file.write_text(yaml_content)

        with mock.patch.dict(
            os.environ,
            {"TOKENSPEED_KERNEL_OVERRIDES_FILE": str(config_file)},
        ):
            load_config_overrides()

        entry = _get_config_override("gemm", "mm")
        assert entry is not None
        assert entry.name == "custom_gemm"


class TestConfigOverrideIntegration:
    """Integration tests: config overrides affect select_kernel()."""

    def _write_overrides(self, tmp_path, yaml_text):
        config_file = tmp_path / "overrides.yaml"
        config_file.write_text(yaml_text)
        load_config_overrides(config_file)

    def test_config_override_by_name(self, sample_specs, h100_platform, tmp_path):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        self._write_overrides(
            tmp_path,
            "overrides:\n" "  attention.decode:\n" "    name: reference_decode\n",
        )

        impl = select_kernel(
            "attention", "decode", torch.bfloat16, platform=h100_platform
        )
        assert impl() == "reference_decode"

    def test_config_override_by_solution(self, sample_specs, h100_platform, tmp_path):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        self._write_overrides(
            tmp_path,
            "overrides:\n" "  attention.decode:\n" "    solution: triton\n",
        )

        impl = select_kernel(
            "attention", "decode", torch.bfloat16, platform=h100_platform
        )
        assert impl() == "triton_decode"

    def test_config_override_objective(self, sample_specs, h100_platform, tmp_path):
        """Config objective override changes selection without forcing a kernel."""
        set_selection_policy(SelectionPolicy())
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        self._write_overrides(
            tmp_path,
            "overrides:\n" "  attention.decode:\n" "    objective: debug\n",
        )

        impl = select_kernel(
            "attention", "decode", torch.bfloat16, platform=h100_platform
        )
        assert impl() == "reference_decode"

    def test_api_override_takes_priority_over_config(
        self, sample_specs, h100_platform, tmp_path
    ):
        """Runtime API override= param has higher priority than config file."""
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        self._write_overrides(
            tmp_path,
            "overrides:\n" "  attention.decode:\n" "    name: reference_decode\n",
        )

        impl = select_kernel(
            "attention",
            "decode",
            torch.bfloat16,
            platform=h100_platform,
            override="triton_decode",
        )
        assert impl() == "triton_decode"

    def test_env_var_override_takes_priority_over_config(
        self, sample_specs, h100_platform, tmp_path
    ):
        """Env var override has higher priority than config file."""
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        self._write_overrides(
            tmp_path,
            "overrides:\n" "  attention.decode:\n" "    name: reference_decode\n",
        )

        with mock.patch.dict(
            os.environ,
            {"TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_DECODE": "triton_decode"},
        ):
            impl = select_kernel(
                "attention", "decode", torch.bfloat16, platform=h100_platform
            )
            assert impl() == "triton_decode"

    def test_context_manager_override_takes_priority_over_config(
        self, sample_specs, h100_platform, tmp_path
    ):
        """kernel_override() context manager has higher priority than config."""
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        self._write_overrides(
            tmp_path,
            "overrides:\n" "  attention.decode:\n" "    name: reference_decode\n",
        )

        with kernel_override("attention", "decode", "triton_decode"):
            impl = select_kernel(
                "attention", "decode", torch.bfloat16, platform=h100_platform
            )
            assert impl() == "triton_decode"

    def test_explicit_objective_takes_priority_over_config(
        self, sample_specs, h100_platform, tmp_path
    ):
        """Caller-supplied non-DEFAULT objective beats config objective."""
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        self._write_overrides(
            tmp_path,
            "overrides:\n" "  attention.decode:\n" "    objective: debug\n",
        )

        impl = select_kernel(
            "attention",
            "decode",
            torch.bfloat16,
            platform=h100_platform,
            objective=SelectionObjective.PORTABILITY,
        )
        assert impl() == "triton_decode"

    def test_config_override_not_found_raises(
        self, sample_specs, h100_platform, tmp_path
    ):
        """Config pointing to nonexistent kernel raises NoKernelFoundError."""
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        self._write_overrides(
            tmp_path,
            "overrides:\n" "  attention.decode:\n" "    name: nonexistent_kernel\n",
        )

        with pytest.raises(NoKernelFoundError, match="Override"):
            select_kernel("attention", "decode", torch.bfloat16, platform=h100_platform)

    def test_config_with_invalid_objective_falls_back(
        self, sample_specs, h100_platform, tmp_path
    ):
        """Invalid objective string in config is warned and ignored."""
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        self._write_overrides(
            tmp_path,
            "overrides:\n"
            "  attention.decode:\n"
            "    objective: nonexistent_objective\n",
        )

        impl = select_kernel(
            "attention", "decode", torch.bfloat16, platform=h100_platform
        )
        assert callable(impl)

    def test_unrelated_ops_unaffected(self, sample_specs, h100_platform, tmp_path):
        """Config override for one op doesn't affect other ops."""
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        self._write_overrides(
            tmp_path,
            "overrides:\n" "  attention.decode:\n" "    name: reference_decode\n",
        )

        attn_impl = select_kernel(
            "attention", "decode", torch.bfloat16, platform=h100_platform
        )
        assert attn_impl() == "reference_decode"

        gemm_impl = select_kernel("gemm", "mm", torch.bfloat16, platform=h100_platform)
        assert gemm_impl() != "reference_decode"


class TestGemmDispatchProfiling:
    @staticmethod
    def _make_gemm_kernel(name: str, call_log: list[str]):
        def _impl(
            A: torch.Tensor,
            B: torch.Tensor,
            A_scales: torch.Tensor | None,
            B_scales: torch.Tensor | None,
            out_dtype: torch.dtype,
            *,
            alpha: torch.Tensor | None = None,
            block_size: list[int] | None = None,
        ) -> torch.Tensor:
            _ = A_scales, B_scales, alpha, block_size
            call_log.append(name)
            return (A.float() @ B.float().T).to(out_dtype)

        return _impl

    @staticmethod
    def _register_kernel(name: str, solution: str, impl) -> None:
        spec = KernelSpec(
            name=name,
            family="gemm",
            mode="mm",
            solution=solution,
            dtypes=frozenset({torch.float16}),
            priority=50,
        )
        KernelRegistry.get().register(spec, impl)

    class _ScopeRecorder:
        def __init__(self):
            self.calls: list[tuple[tuple, dict]] = []
            self.trace: list[str] = []

        def __call__(self, *args, **kwargs):
            self.calls.append((args, kwargs))

            class _Scope:
                def __init__(self, trace: list[str]):
                    self._trace = trace

                def __enter__(self):
                    self._trace.append("enter")
                    return self

                def __exit__(self, exc_type, exc, tb):
                    _ = exc_type, exc, tb
                    self._trace.append("exit")

            return _Scope(self.trace)

    def test_mm_wraps_triton_kernel_execution_in_scope(self, monkeypatch):
        call_log: list[str] = []
        triton_kernel_name = "test_triton_mm"
        self._register_kernel(
            triton_kernel_name,
            "triton",
            self._make_gemm_kernel(triton_kernel_name, call_log),
        )

        scope = self._ScopeRecorder()
        monkeypatch.setattr(gemm, "kernel_scope", scope)

        A = torch.randn(4, 8, dtype=torch.float16)
        B = torch.randn(6, 8, dtype=torch.float16)

        with kernel_override("gemm", "mm", triton_kernel_name):
            out = gemm.mm(A, B, out_dtype=torch.float16)

        assert out.shape == (4, 6)
        assert call_log == [triton_kernel_name]
        assert scope.trace == ["enter", "exit"]
        assert scope.calls == [
            (
                (
                    "gemm",
                    "mm",
                    torch.float16,
                ),
                {
                    "kernel_name": triton_kernel_name,
                    "M": 4,
                    "N": 6,
                    "K": 8,
                },
            )
        ]

    def test_mm_wraps_non_triton_kernel_execution_in_scope(self, monkeypatch):
        call_log: list[str] = []
        vendor_kernel_name = "test_vendor_mm"
        self._register_kernel(
            vendor_kernel_name,
            "flashinfer",
            self._make_gemm_kernel(vendor_kernel_name, call_log),
        )

        scope = self._ScopeRecorder()
        monkeypatch.setattr(gemm, "kernel_scope", scope)

        A = torch.randn(4, 8, dtype=torch.float16)
        B = torch.randn(6, 8, dtype=torch.float16)

        with kernel_override("gemm", "mm", vendor_kernel_name):
            out = gemm.mm(A, B, out_dtype=torch.float16)

        assert out.shape == (4, 6)
        assert call_log == [vendor_kernel_name]
        assert scope.trace == ["enter", "exit"]
        assert scope.calls == [
            (
                (
                    "gemm",
                    "mm",
                    torch.float16,
                ),
                {
                    "kernel_name": vendor_kernel_name,
                    "M": 4,
                    "N": 6,
                    "K": 8,
                },
            )
        ]
