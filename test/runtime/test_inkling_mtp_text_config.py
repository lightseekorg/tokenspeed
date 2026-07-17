"""Inkling MTP draft text-config derivation.

Checkpoints since 4d71c3ea mix SWA and full-attention MTP depths
(``mtp_config.local_layer_ids``); ``inkling_mtp_text_config`` turns the base
text config into the draft worker's depth config: depth-local ids become the
``local_layer_ids`` that drive attention geometry and paged-cache labels,
and depths beyond ``--speculative-num-steps`` are pruned (an MTP chain only
runs depths 0..steps-1).
"""

# CI Registration (parsed via AST, runtime no-op)
import os
import sys
import unittest

from tokenspeed.runtime.configs.inkling_config import (
    InklingMMConfig,
    inkling_kv_heads_for_layer,
    inkling_mtp_text_config,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci
from runtime.models.inkling_fixtures import TINY_MM_CONFIG

register_cuda_ci(est_time=5, suite="runtime-1gpu")


def _mm_config(mtp_config):
    cfg_dict = {
        k: v
        for k, v in {**TINY_MM_CONFIG, "mtp_config": mtp_config}.items()
        if k not in ("model_type", "architectures")
    }
    return InklingMMConfig(**cfg_dict)


class TestInklingMTPTextConfig(unittest.TestCase):
    def test_mtp_local_ids_parsed_from_mtp_config(self):
        cfg = _mm_config(
            {"num_nextn_predict_layers": 8, "local_layer_ids": [0, 2, 4, 5, 6, 7]}
        )
        self.assertEqual(cfg.get_text_config().mtp_local_layer_ids, [0, 2, 4, 5, 6, 7])

    def test_mtp_local_ids_validation(self):
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            _mm_config({"num_nextn_predict_layers": 3, "local_layer_ids": [0, 3]})
        with self.assertRaisesRegex(ValueError, "duplicates"):
            _mm_config({"num_nextn_predict_layers": 3, "local_layer_ids": [0, 0]})

    def test_depth_config_prunes_to_steps_and_keeps_local_ids(self):
        text = _mm_config(
            {"num_nextn_predict_layers": 8, "local_layer_ids": [0, 2, 4, 5, 6, 7]}
        ).get_text_config()
        cfg = inkling_mtp_text_config(text, num_steps=3)
        self.assertEqual(cfg.num_hidden_layers, 3)
        self.assertEqual(cfg.num_nextn_predict_layers, 3)
        self.assertEqual(cfg.local_layer_ids, [0, 2])
        self.assertEqual(cfg.dense_mlp_idx, 3)
        # Depth 1 is full attention at the ckpt head count; depths 0/2 are
        # SWA at the swa count — the target model's byte-uniform pairing.
        heads = [inkling_kv_heads_for_layer(cfg, i, True) for i in range(3)]
        self.assertEqual(
            heads,
            [
                cfg.swa_num_key_value_heads,
                cfg.ckpt_num_key_value_heads,
                cfg.swa_num_key_value_heads,
            ],
        )
        # Every depth gets its own paged-cache group so the shared slab has
        # no dead rows (1 full + 2 sliding sub-groups of one layer each).
        self.assertEqual(
            cfg.paged_cache_layer_types,
            ["sliding_attention_0", "full_attention", "sliding_attention_1"],
        )
        # The base config is untouched (deepcopy).
        self.assertEqual(
            text.num_hidden_layers, TINY_MM_CONFIG["text_config"]["num_hidden_layers"]
        )

    def test_depth_config_idempotent(self):
        text = _mm_config(
            {"num_nextn_predict_layers": 8, "local_layer_ids": [0, 2]}
        ).get_text_config()
        cfg = inkling_mtp_text_config(text, num_steps=3)
        self.assertIs(inkling_mtp_text_config(cfg), cfg)
        self.assertIs(inkling_mtp_text_config(cfg, num_steps=1), cfg)

    def test_no_steps_keeps_all_depths(self):
        text = _mm_config(
            {"num_nextn_predict_layers": 4, "local_layer_ids": []}
        ).get_text_config()
        cfg = inkling_mtp_text_config(text)
        self.assertEqual(cfg.num_hidden_layers, 4)
        self.assertEqual(cfg.local_layer_ids, [])
        self.assertEqual(cfg.paged_cache_layer_types, ["full_attention"] * 4)

    def test_steps_beyond_depths_keeps_all_depths(self):
        # The registry raises for steps > depths; the config transform must
        # not mask that by growing the depth count.
        text = _mm_config(
            {"num_nextn_predict_layers": 4, "local_layer_ids": []}
        ).get_text_config()
        cfg = inkling_mtp_text_config(text, num_steps=9)
        self.assertEqual(cfg.num_hidden_layers, 4)


if __name__ == "__main__":
    unittest.main()
