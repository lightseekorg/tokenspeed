"""KV profile -> layout-grid flooring.

The memory-dependent profile can land anywhere on the base page grid, but
the hetero slot layout is only viewable on a coarser congruence. The MTP
draft pool shares the target's page-id space at the largest group page's
stride (one 256-row slot per id plus the 256-row dummy for the Inkling
geometry — the drafter consumes the flat full-attention table at its
native stride), so the profiled size floors to ``size ≡ 0 (mod 256)``.
Historical: before the drafter consumed the table at that stride, the
draft pool held one 128-row slot per id and its 256-row page view forced
``size ≡ 128 (mod 256)`` (an ODD id count) — whether a boot survived used
to be the parity of the profiled id count.
"""

# CI Registration (parsed via AST, runtime no-op)
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from tokenspeed.runtime.layers.attention.registry import (
    _cache_storage_report,
    _floor_tokens_to_layout_grid,
    _inkling_checkpoint_group_specs,
    _inkling_lcm_fields,
    _max_lcm_packing,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")


def _hetero_config():
    return SimpleNamespace(
        layer_kv_head_counts=(16,) * 55 + (8,) * 11,
        slot_tokens=256,
        group_page_sizes={"full_attention": 256},
        page_size=128,
    )


class TestKvProfileLayoutGrid(unittest.TestCase):
    def test_bad_boot_value_lands_on_grid(self):
        # The 2026-07-14 crash value (even id count, 5758) now floors onto
        # the 256 grid instead of the odd-parity grid.
        floored = _floor_tokens_to_layout_grid(737024, _hetero_config())
        self.assertEqual(floored, 737024)
        self.assertEqual(floored % 256, 0)

    def test_off_grid_value_floors(self):
        floored = _floor_tokens_to_layout_grid(737024 + 128, _hetero_config())
        self.assertEqual(floored, 737024)
        self.assertEqual(floored % 256, 0)

    def test_every_residue_lands_on_grid_and_costs_under_one_page(self):
        cfg = _hetero_config()
        for raw in range(736896, 736896 + 512, 128):
            floored = _floor_tokens_to_layout_grid(raw, cfg)
            self.assertLessEqual(floored, raw)
            self.assertLess(raw - floored, 256)
            self.assertEqual(floored % 256, 0)

    def test_noop_without_hetero_head_counts(self):
        cfg = SimpleNamespace(
            layer_kv_head_counts=None,
            slot_tokens=None,
            group_page_sizes=None,
            page_size=128,
        )
        self.assertEqual(_floor_tokens_to_layout_grid(737024, cfg), 737024)


class TestCacheStorageReport(unittest.TestCase):
    def test_lcm_report_uses_arena_and_geometry(self):
        plan = SimpleNamespace(
            arena_bytes=9_000,
            logical_block_tokens=128,
            num_lcm_blocks=4,
            groups=(
                SimpleNamespace(
                    group_id="full_attention",
                    cache_blocks_per_lcm_block=16,
                ),
                SimpleNamespace(
                    group_id="linear_attention",
                    cache_blocks_per_lcm_block=1,
                ),
            ),
        )
        pool = SimpleNamespace(
            size=8_192,
            page_size=128,
            _lcm_memory_plan=plan,
            _lcm_arena=SimpleNamespace(backing=SimpleNamespace(nbytes=9_000)),
        )
        draft_pool = SimpleNamespace(
            get_kv_size_bytes=lambda: (100, 200),
        )

        report = _cache_storage_report(
            configured_cache_bytes=10_000,
            pool=pool,
            draft_pool=draft_pool,
            mamba_pool=None,
        )

        self.assertEqual(report["allocated_cache_bytes"], 9_300)
        self.assertEqual(report["physical_token_capacity"], 8_192)
        self.assertEqual(report["capacity_source"], "lcm_geometry")
        self.assertEqual(report["geometry"]["num_lcm_blocks"], 4)
        self.assertEqual(
            report["geometry"]["cache_blocks_per_lcm_block"],
            {"full_attention": 16, "linear_attention": 1},
        )

    def test_radix_report_counts_kv_and_mamba_allocations(self):
        pool = SimpleNamespace(
            size=4_096,
            page_size=128,
            _slot_tokens=128,
            _lcm_memory_plan=None,
            k_buffer=(SimpleNamespace(shape=(4_224,)),),
            get_kv_size_bytes=lambda: (1_000, 2_000),
        )
        mamba_pool = SimpleNamespace(
            conv_state=SimpleNamespace(nbytes=300),
            ssm_state=SimpleNamespace(nbytes=400),
        )

        report = _cache_storage_report(
            configured_cache_bytes=10_000,
            pool=pool,
            draft_pool=None,
            mamba_pool=mamba_pool,
        )

        self.assertEqual(report["allocated_cache_bytes"], 3_700)
        self.assertEqual(report["physical_token_capacity"], 4_096)
        self.assertEqual(report["capacity_source"], "allocated_token_rows")


class TestInklingCheckpointGroupSpecs(unittest.TestCase):
    def test_draft_capacity_uses_attention_packing_not_checkpoint_packing(self):
        plan = SimpleNamespace(
            groups=(
                SimpleNamespace(
                    group_id="full_attention",
                    cache_blocks_per_lcm_block=2,
                ),
                SimpleNamespace(
                    group_id="sliding_attention_0",
                    cache_blocks_per_lcm_block=1,
                ),
                SimpleNamespace(
                    group_id="kvconv",
                    cache_blocks_per_lcm_block=21,
                ),
                SimpleNamespace(
                    group_id="hiddenconv",
                    cache_blocks_per_lcm_block=2,
                ),
            )
        )

        self.assertEqual(
            _max_lcm_packing(
                plan,
                {"full_attention", "sliding_attention_0"},
            ),
            2,
        )
        self.assertEqual(_max_lcm_packing(plan), 21)

    def test_model_mapping_defaults_to_bf16_checkpoint_columns(self):
        attention_groups = (
            "sliding_attention_0",
            "sliding_attention_1",
            "sliding_attention_2",
            "sliding_attention_3",
            "sliding_attention_4",
            "full_attention",
        )
        config = SimpleNamespace(
            layer_types=attention_groups * 2,
            layer_kv_head_counts=(16, 16, 16, 16, 16, 8) * 2,
            attn_tp_size=8,
            head_dim=128,
            kv_cache_dtype=SimpleNamespace(itemsize=1),
            kv_cache_mxfp8=True,
        )
        text_config = SimpleNamespace(hidden_size=4096, sconv_kernel_size=4)

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("INKLING_FP8_SCONV", None)
            fields = _inkling_lcm_fields(config, text_config, 128)

        self.assertEqual(
            {field.group_id for field in fields},
            {*attention_groups, "kvconv", "hiddenconv"},
        )
        self.assertEqual(
            next(field for field in fields if field.field_id == "layer.0.k").shape,
            (128, 2, 128),
        )
        self.assertEqual(
            next(field for field in fields if field.field_id == "layer.5.k").shape,
            (128, 1, 128),
        )
        self.assertEqual(
            next(
                field for field in fields if field.field_id == "layer.0.attnconv"
            ).element_size,
            2,
        )

    def test_model_mapping_allows_explicit_fp8_checkpoint_columns(self):
        config = SimpleNamespace(
            layer_types=("sliding_attention_0", "full_attention"),
            layer_kv_head_counts=(16, 8),
            attn_tp_size=8,
            head_dim=128,
            kv_cache_dtype=SimpleNamespace(itemsize=1),
            kv_cache_mxfp8=True,
        )
        text_config = SimpleNamespace(hidden_size=4096, sconv_kernel_size=4)

        with mock.patch.dict(os.environ, {"INKLING_FP8_SCONV": "1"}):
            fields = _inkling_lcm_fields(config, text_config, 128)

        self.assertEqual(
            next(
                field for field in fields if field.field_id == "layer.0.attnconv"
            ).element_size,
            1,
        )

    def test_two_state_groups_share_the_domain_block_size(self):
        plan = SimpleNamespace(
            logical_block_tokens=128,
            groups=(
                SimpleNamespace(
                    group_id="full_attention",
                    cache_blocks_per_lcm_block=2,
                ),
                SimpleNamespace(
                    group_id="sliding_attention_0",
                    cache_blocks_per_lcm_block=1,
                ),
                SimpleNamespace(
                    group_id="kvconv",
                    cache_blocks_per_lcm_block=21,
                ),
                SimpleNamespace(
                    group_id="hiddenconv",
                    cache_blocks_per_lcm_block=2,
                ),
            ),
        )

        specs = _inkling_checkpoint_group_specs(plan)

        self.assertEqual([spec.group_id for spec in specs], ["kvconv", "hiddenconv"])
        self.assertEqual(
            [spec.cache_blocks_per_lcm_block for spec in specs],
            [21, 2],
        )
        for spec in specs:
            self.assertEqual(spec.rows_per_page, 128)
            self.assertEqual(spec.entry_stride_tokens, 1)
            self.assertEqual(spec.family, "state")
            self.assertEqual(spec.retention, "full_history")
            self.assertIsNone(spec.sliding_window_tokens)
            self.assertIsNone(spec.block_size)
            self.assertTrue(spec.materializes_all_boundaries)


if __name__ == "__main__":
    unittest.main()
