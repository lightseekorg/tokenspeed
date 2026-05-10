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

from tokenspeed.runtime.layers.moe.backends.mxfp4.weights import create_mxfp4_weights


class _Backend:
    def _make_weight_loader(self):
        def _weight_loader(*args, **kwargs):
            del args, kwargs

        return _weight_loader


class TestMxfp4Weights(unittest.TestCase):
    def test_scale_weights_store_checkpoint_bytes(self):
        layer = nn.Module()
        create_mxfp4_weights(
            _Backend(),
            layer,
            num_local_experts=2,
            hidden_size_padded=64,
            ispp_padded=96,
        )

        self.assertEqual(layer.w13_weight_scale.dtype, torch.uint8)
        self.assertEqual(layer.w2_weight_scale.dtype, torch.uint8)


if __name__ == "__main__":
    unittest.main()
