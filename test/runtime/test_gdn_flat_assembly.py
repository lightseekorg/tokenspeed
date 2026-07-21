"""Flat assembly line for GDN hybrids (M17 C4).

Three contracts: the Qwen3.5 config exposes ``layer_types`` in the
paged-cache label vocabulary; the page-size equalization decision
(``equalized_block_size``) inflates P to cover the GDN state row;
and an MHAConfig carrying state shapes builds a full-coverage pool with
one (conv, ssm) slab pair per state layer, both cache groups published,
and the ctor geometry check enforcing the equalized P. Flat GDN sizing
itself is plan-driven (plan_component_tensors, test_flat_memory_plan).
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import unittest
from unittest import mock

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="runtime-1gpu")

_CONFIGS_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "python"
    / "tokenspeed"
    / "runtime"
    / "configs"
)

_PKG_FLAT_PROBE = (
    "tokenspeed.runtime.configs.paged_cache_spec.scheduler_ext_flat_kvcache"
)


def _load(mod_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, _CONFIGS_DIR / file_name)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: on py3.9 @dataclass + `from __future__ import
    # annotations` resolves field types via sys.modules[cls.__module__].
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_plan = _load("flat_memory_plan_gdn_assembly_under_test", "flat_memory_plan.py")
equalized_block_size = _plan.equalized_block_size


# Qwen3.5-ish interleaving: 3 linear layers then 1 full, times 12 (48 layers).
QWEN3_5ISH_LAYER_TYPES = (["linear_attention"] * 3 + ["full_attention"]) * 12

# Qwen3.5 defaults at attn TP=1 (configs/qwen3_5_text_base_config.py):
# KV row: 2 (K+V) * 2 kv heads * 256 head_dim * 2 B (bf16) per token-layer.
QWEN3_5ISH_KV_BYTES_PER_SLOT = 2 * 2 * 256 * 2  # 2048
# conv: (2*128*16 + 128*32) x (4 - 1) in bf16; ssm: 32 x 128 x 128 in fp32.
QWEN3_5ISH_CONV_BYTES = (2 * 128 * 16 + 128 * 32) * 3 * 2  # 49152
QWEN3_5ISH_SSM_BYTES = 32 * 128 * 128 * 4  # 2097152


class Qwen3_5LayerTypesTest(unittest.TestCase):
    """The config's layer_types property (interleaving + label vocabulary)."""

    def setUp(self):
        try:
            from tokenspeed.runtime.configs.qwen3_5_text_base_config import (
                Qwen3_5BaseTextConfig,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + transformers: {exc}")
        self.config_cls = Qwen3_5BaseTextConfig

    def test_layer_types_interleaving(self):
        cfg = self.config_cls(num_hidden_layers=8, full_attention_interval=4)
        self.assertEqual(
            cfg.layer_types,
            (["linear_attention"] * 3 + ["full_attention"]) * 2,
        )

    def test_layers_block_type_keeps_checkpoint_label(self):
        # models/qwen3_5.py keys layer construction on the checkpoint's
        # "attention" label; layer_types must not change it.
        cfg = self.config_cls(num_hidden_layers=4, full_attention_interval=4)
        self.assertEqual(
            cfg.layers_block_type, ["linear_attention"] * 3 + ["attention"]
        )

    def test_tracks_nextn_interval_override(self):
        # models/qwen3_5_nextn.py rewrites full_attention_interval AFTER
        # construction; layer_types must follow (property, not __init__).
        cfg = self.config_cls(num_hidden_layers=2, full_attention_interval=4)
        cfg.full_attention_interval = 1
        self.assertEqual(cfg.layer_types, ["full_attention"] * 2)

    def test_temporal_state_shape_is_native_k_last(self):
        from tokenspeed.runtime.utils.env import global_server_args_dict

        mapping = mock.Mock()
        mapping.attn.tp_size = 2
        cfg = self.config_cls(
            full_attention_interval=4,
            linear_num_value_heads=8,
            linear_key_head_dim=3,
            linear_value_head_dim=5,
        )
        with mock.patch.dict(global_server_args_dict, {"mapping": mapping}):
            temporal_state_shape = cfg.mamba2_cache_params[1]

        self.assertEqual(temporal_state_shape, (4, 5, 3))


class EqualizedPageSizeTest(unittest.TestCase):
    """Pure equalization decision (no torch)."""

    def _equalized(self, block_size, **kwargs):
        return equalized_block_size(
            layer_types=QWEN3_5ISH_LAYER_TYPES,
            kv_bytes_per_slot=QWEN3_5ISH_KV_BYTES_PER_SLOT,
            state_const_bytes={
                "conv": QWEN3_5ISH_CONV_BYTES,
                "ssm": QWEN3_5ISH_SSM_BYTES,
            },
            block_size=block_size,
            **kwargs,
        )

    def test_inflates_to_cover_state_row(self):
        # ceil((49152 + 2097152) / 2048) = 1048 tokens to cover the state
        # row; default alignment = original block size 64 -> 1088.
        self.assertEqual(self._equalized(64), 1088)

    def test_unchanged_when_kv_row_already_covers(self):
        self.assertEqual(self._equalized(2048), 2048)

    def test_explicit_alignment(self):
        # 1048 rounded up to a multiple of 16 -> 1056.
        self.assertEqual(self._equalized(64, alignment=16), 1056)

    def test_no_state_layers_is_identity(self):
        self.assertEqual(
            equalized_block_size(
                layer_types=["full_attention"] * 4,
                kv_bytes_per_slot=QWEN3_5ISH_KV_BYTES_PER_SLOT,
                state_const_bytes={},
                block_size=64,
            ),
            64,
        )


class GdnFlatPoolAssemblyTest(unittest.TestCase):
    """MHAConfig with state shapes -> create_pool: full-coverage pool with
    state slabs, both published groups, and the equalized-P geometry gate."""

    # 3 linear + 1 full; kv row = 2 * 1 head * 8 dim * 2 B = 32 B/slot;
    # state row = conv (4*3 bf16 = 24 B) + ssm (2*4*4 fp32 = 128 B) = 152 B;
    # ceil(152 / 32) = 5 > block 4 -> equalized P = 8 (multiple of 4).
    LAYER_TYPES = ("linear_attention",) * 3 + ("full_attention",)
    CONV_SHAPE = (4, 3)
    TEMPORAL_SHAPE = (2, 4, 4)

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.configs.mha import (
                MHAConfig,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.MHAConfig = MHAConfig

    def _config(self, page_size: int):
        torch = self.torch
        return self.MHAConfig(
            device="cpu",
            backend_name=None,
            num_attention_heads=2,
            num_kv_heads=1,
            head_dim=8,
            attn_tp_size=1,
            dtype=torch.bfloat16,
            kv_cache_dtype=torch.bfloat16,
            page_size=page_size,
            context_len=64,
            max_bs=2,
            max_graph_bs=2,
            kv_cache_quant_method=None,
            layer_types=self.LAYER_TYPES,
            max_scheduled_tokens=16,
            conv_state_shape=self.CONV_SHAPE,
            temporal_state_shape=self.TEMPORAL_SHAPE,
            conv_dtype=torch.bfloat16,
            ssm_dtype=torch.float32,
        )

    def _pool(self, page_size: int):
        with mock.patch(_PKG_FLAT_PROBE, return_value=True):
            return self._config(page_size).create_pool(
                len(self.LAYER_TYPES), 32, 0, False
            )

    def test_equalization_decision_matches_pure_helper(self):
        self.assertEqual(
            equalized_block_size(
                layer_types=list(self.LAYER_TYPES),
                kv_bytes_per_slot=32,
                state_const_bytes={"conv": 24, "ssm": 128},
                block_size=4,
            ),
            8,
        )

    def test_assembly_at_equalized_page_size(self):
        pool = self._pool(page_size=8)
        # One (conv, ssm) slab pair per state layer, rows over the shared
        # page-id space (size // P + 1 null row).
        self.assertEqual(len(pool.state_slabs), 3)
        conv, ssm = pool.state_slabs[0]
        self.assertEqual(tuple(conv.shape), (5, *self.CONV_SHAPE))
        self.assertEqual(conv.dtype, self.torch.bfloat16)
        self.assertEqual(tuple(ssm.shape), (5, *self.TEMPORAL_SHAPE))
        self.assertEqual(ssm.dtype, self.torch.float32)
        # Both groups published (upstream signal for flat state paging).
        self.assertEqual(
            sorted(spec.group_id for spec in pool.paged_cache_group_specs),
            ["full_attention", "linear_attention"],
        )
        # Plan-sized coverage (M18a T4): the k/v lists stay layer-indexed,
        # but state layers carry no KV tensors (None slots) -- only the
        # full-attention layer allocates.
        self.assertEqual(len(pool.k_buffer), len(self.LAYER_TYPES))
        for layer_id, label in enumerate(self.LAYER_TYPES):
            if label == "linear_attention":
                self.assertIsNone(pool.k_buffer[layer_id])
                self.assertIsNone(pool.v_buffer[layer_id])
            else:
                self.assertIsNotNone(pool.k_buffer[layer_id])
                self.assertIsNotNone(pool.v_buffer[layer_id])

    def test_geometry_raises_at_original_page_size(self):
        with self.assertRaisesRegex(ValueError, "pre-equalized"):
            self._pool(page_size=4)


if __name__ == "__main__":
    unittest.main()
