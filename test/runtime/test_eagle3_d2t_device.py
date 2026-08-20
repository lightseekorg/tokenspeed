"""Regression test: EAGLE3 ``d2t`` loading follows the weight's device.

The draft vocabulary map is built as ``loaded_weight + torch.arange(n)``.
``torch.arange`` defaults to CPU, which matched every CPU-staging weight
loader and so went unnoticed -- until a GPU-direct loader
(``--load-format instanttensor``) started yielding ``loaded_weight`` already
on the device, making the addition raise "Expected all tensors to be on the
same device".
"""

import os
import sys
import types
import unittest
from unittest import mock

import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.models.deepseek_v3 import (
    DeepseekV3ForCausalLM,
    Eagle3DeepseekV2ForCausalLM,
)
from tokenspeed.runtime.models.llama_eagle3 import LlamaForCausalLMEagle3


@unittest.skipIf(not torch.cuda.is_available(), "needs a GPU to place the weight on")
class TestEagle3D2TDevice(unittest.TestCase):
    D2T = torch.tensor([5, 7, 9], dtype=torch.long)

    def _assert_hot_token_id(self, model):
        # loaded_weight + arange(3) == [5, 8, 11]
        self.assertEqual(model.hot_token_id.device.type, "cuda")
        self.assertTrue(
            torch.equal(
                model.hot_token_id.cpu(), torch.tensor([5, 8, 11], dtype=torch.long)
            )
        )

    def test_llama_eagle3_places_hot_token_id_on_the_weight_device(self):
        model = types.SimpleNamespace(named_parameters=lambda: iter([]))
        LlamaForCausalLMEagle3.load_weights(model, [("d2t", self.D2T.to("cuda"))])
        self._assert_hot_token_id(model)

    def test_deepseek_eagle3_places_hot_token_id_on_the_weight_device(self):
        # Only the ``d2t`` branch is under test. The instance is built without
        # ``__init__`` so zero-arg ``super()`` still resolves, and the base
        # class -- which consumes the remaining (here empty) list and needs a
        # real module tree -- is stubbed out.
        model = object.__new__(Eagle3DeepseekV2ForCausalLM)
        with mock.patch.object(DeepseekV3ForCausalLM, "load_weights"):
            Eagle3DeepseekV2ForCausalLM.load_weights(
                model, [("d2t", self.D2T.to("cuda"))]
            )
        self._assert_hot_token_id(model)


if __name__ == "__main__":
    unittest.main()
