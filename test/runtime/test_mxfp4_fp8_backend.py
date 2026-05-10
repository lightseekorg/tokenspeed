"""Unit tests for the FP8 x MXFP4 MoE backend (AMD Quark ``w_mxfp4_a_fp8``)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

# NOTE: ``tokenspeed_kernel`` (which loads ``tokenspeed_triton``'s C
# extension) must be imported *before* ``torch`` to avoid a torch / triton
# ABI mismatch that segfaults ``libtriton.so`` initialisation when torch's
# allocator is loaded first. Mirror the workaround in
# ``tokenspeed.runtime.utils.common``.
import tokenspeed_kernel  # noqa: E402, F401
import torch  # noqa: E402

from tokenspeed.runtime.layers.moe.backends.mxfp4.triton_kernel_fp8 import (  # noqa: E402
    _per_tensor_input_scale_loader,
    create_mxfp4_fp8_input_scales,
)
from tokenspeed.runtime.layers.quantization.mxfp4 import (  # noqa: E402
    Mxfp4Config,
    _is_quark_w_mxfp4_a_fp8,
)


_AMD_QUARK_CFG = {
    "quant_method": "quark",
    "global_quant_config": {
        "weight": {
            "dtype": "fp4",
            "group_size": 32,
            "qscheme": "per_group",
            "scale_format": "e8m0",
        },
        "input_tensors": {
            "dtype": "fp8_e4m3",
            "qscheme": "per_tensor",
            "is_dynamic": False,
        },
    },
    "exclude": ["lm_head", "model.layers.0.self_attn.q_proj"],
}

_OAI_MXFP4_CFG = {"quant_method": "mxfp4"}


class TestQuarkDetection(unittest.TestCase):
    def test_detects_amd_quark_w_mxfp4_a_fp8(self):
        self.assertTrue(_is_quark_w_mxfp4_a_fp8(_AMD_QUARK_CFG))

    def test_rejects_oai_mxfp4(self):
        self.assertFalse(_is_quark_w_mxfp4_a_fp8(_OAI_MXFP4_CFG))

    def test_rejects_quark_without_fp8_act(self):
        cfg = {
            "quant_method": "quark",
            "global_quant_config": {
                "weight": {"dtype": "fp4", "group_size": 32},
                "input_tensors": {"dtype": "bf16"},
            },
        }
        self.assertFalse(_is_quark_w_mxfp4_a_fp8(cfg))

    def test_override_promotes_to_mxfp4(self):
        self.assertEqual(
            Mxfp4Config.override_quantization_method(_AMD_QUARK_CFG, None),
            "mxfp4",
        )
        self.assertIsNone(
            Mxfp4Config.override_quantization_method(_OAI_MXFP4_CFG, None)
        )

    def test_from_config_sets_w4a8_fp8_flag(self):
        cfg = Mxfp4Config.from_config(_AMD_QUARK_CFG)
        self.assertTrue(cfg.is_w4a8_fp8)
        self.assertTrue(cfg.is_checkpoint_mxfp4_serialized)
        self.assertEqual(len(cfg.excluded_layers), 2)

    def test_from_config_oai_does_not_set_flag(self):
        cfg = Mxfp4Config.from_config(_OAI_MXFP4_CFG)
        self.assertFalse(cfg.is_w4a8_fp8)
        self.assertTrue(cfg.is_checkpoint_mxfp4_serialized)


class TestInputScaleLoader(unittest.TestCase):
    def test_create_input_scale_params(self):
        layer = torch.nn.Module()
        create_mxfp4_fp8_input_scales(layer, num_local_experts=4)
        self.assertEqual(layer.w13_input_scale.shape, (4,))
        self.assertEqual(layer.w2_input_scale.shape, (4,))
        self.assertEqual(layer.w13_input_scale.dtype, torch.float32)
        self.assertEqual(layer.w2_input_scale.dtype, torch.float32)
        self.assertTrue(hasattr(layer.w13_input_scale, "weight_loader"))
        self.assertTrue(hasattr(layer.w2_input_scale, "weight_loader"))

    def test_w13_loader_keeps_max(self):
        layer = torch.nn.Module()
        create_mxfp4_fp8_input_scales(layer, num_local_experts=2)
        param = layer.w13_input_scale
        loader = param.weight_loader
        loader(param, torch.tensor(0.5), shard_id="w1", local_expert_id=0)
        loader(param, torch.tensor(0.7), shard_id="w3", local_expert_id=0)
        loader(param, torch.tensor(0.1), shard_id="w1", local_expert_id=1)
        self.assertAlmostEqual(float(param.data[0]), 0.7, places=5)
        self.assertAlmostEqual(float(param.data[1]), 0.1, places=5)

    def test_w2_loader_overwrites(self):
        layer = torch.nn.Module()
        create_mxfp4_fp8_input_scales(layer, num_local_experts=3)
        param = layer.w2_input_scale
        loader = param.weight_loader
        loader(param, torch.tensor(0.25), shard_id="w2", local_expert_id=1)
        self.assertAlmostEqual(float(param.data[1]), 0.25, places=5)
        self.assertEqual(float(param.data[0]), 0.0)
        self.assertEqual(float(param.data[2]), 0.0)

    def test_loader_rejects_unknown_shard(self):
        layer = torch.nn.Module()
        create_mxfp4_fp8_input_scales(layer, num_local_experts=1)
        param = layer.w13_input_scale
        with self.assertRaises(ValueError):
            _per_tensor_input_scale_loader(
                param, torch.tensor(1.0), shard_id="bogus", local_expert_id=0
            )


if __name__ == "__main__":
    unittest.main()
