"""Regression tests for shared MXFP4 MoE weight allocation."""

import os
import sys
import unittest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

import torch
from torch import nn

from tokenspeed.runtime.layers.moe.types import MoELayerSpec
from tokenspeed.runtime.layers.moe.weights.loaders import load_model_weight
from tokenspeed.runtime.layers.moe.weights.mxfp4 import (
    create_mxfp4_weight_pair,
)


def _mxfp4_spec(num_local_experts: int, hidden_size: int, intermediate_size: int):
    return MoELayerSpec(
        top_k=2,
        num_experts=num_local_experts,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="silu",
        tp_rank=0,
        tp_size=1,
        ep_rank=0,
        ep_size=1,
    )


class TestMxfp4Weights(unittest.TestCase):
    def test_scale_weights_store_checkpoint_bytes(self):
        layer = nn.Module()
        create_mxfp4_weight_pair(
            _mxfp4_spec(
                num_local_experts=2,
                hidden_size=64,
                intermediate_size=96,
            ),
            layer,
        )

        self.assertEqual(layer.w13_weight_scale.dtype, torch.uint8)
        self.assertEqual(layer.w2_weight_scale.dtype, torch.uint8)

    def test_e8m0_scale_load_preserves_checkpoint_bytes(self):
        e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
        if e8m0_dtype is None:
            self.skipTest("torch.float8_e8m0fnu is unavailable")

        layer = nn.Module()
        create_mxfp4_weight_pair(
            _mxfp4_spec(
                num_local_experts=1,
                hidden_size=64,
                intermediate_size=64,
            ),
            layer,
        )

        raw_w1_scale = (
            torch.tensor([120, 121], dtype=torch.uint8).repeat(64).reshape(64, 2)
        )
        load_model_weight(
            layer.w13_weight_scale,
            raw_w1_scale.view(e8m0_dtype),
            "w1",
            local_expert_id=0,
            tp_rank=0,
            is_bias=False,
            use_presharded_weights=False,
            do_transpose=False,
        )
        self.assertTrue(torch.equal(layer.w13_weight_scale.data[0, :64], raw_w1_scale))

        raw_w2_scale = (
            torch.tensor([122, 123], dtype=torch.uint8).repeat(64).reshape(64, 2)
        )
        load_model_weight(
            layer.w2_weight_scale,
            raw_w2_scale.view(e8m0_dtype),
            "w2",
            local_expert_id=0,
            tp_rank=0,
            is_bias=False,
            use_presharded_weights=False,
            do_transpose=False,
        )
        self.assertTrue(torch.equal(layer.w2_weight_scale.data[0], raw_w2_scale))


if __name__ == "__main__":
    unittest.main()
