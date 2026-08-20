"""Regression test: GPT-OSS MXFP4 weight loading streams the iterator.

A GPU-direct loader (``--load-format instanttensor``) yields each checkpoint
tensor already resident on the GPU. ``_load_mxfp4_weights`` must therefore
consume the weight iterator lazily and copy each (large) MoE expert tensor
straight into its slot, rather than buffering every expert tensor into a list
first -- the latter keeps the whole checkpoint on the device at once and OOMs
mid-load. This test pins that behavior without needing a GPU or real weights.
"""

import os
import sys
import types
import unittest

import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.models.gpt_oss import GptOssForCausalLM


class TestGptOssMxfp4Streaming(unittest.TestCase):
    def test_load_mxfp4_weights_streams_experts(self):
        # The "weight" stand-in is the name; the stubs never touch tensor data.
        items = [
            "model.embed_tokens.weight",
            "model.layers.0.mlp.experts.gate_up_proj_blocks",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.mlp.experts.down_proj_blocks",
            "lm_head.weight",
        ]

        pulled = []

        def source():
            for name in items:
                pulled.append(name)
                yield name, name

        seen_experts = []
        received = {}

        def fake_load_experts(weights):
            received["is_generator"] = isinstance(weights, types.GeneratorType)
            iterator = iter(weights)
            first_expert = next(iterator)
            seen_experts.append(first_expert[0])
            # Item 2 of 5: reaching the first expert must not drain the source.
            received["pulled_after_first_expert"] = len(pulled)
            for name, _ in iterator:
                seen_experts.append(name)
            return {"loaded_expert_param"}

        normal_seen = {}

        def fake_load_normal(
            normal_weights, *, weight_name_mapping, other_loaded_param_names
        ):
            normal_seen["names"] = [name for name, _ in normal_weights]
            normal_seen["other"] = other_loaded_param_names

        fake_self = types.SimpleNamespace(
            _load_mxfp4_experts_weights=fake_load_experts,
            _load_normal_weights=fake_load_normal,
        )

        GptOssForCausalLM._load_mxfp4_weights(
            fake_self, source(), weight_name_mapping={}
        )

        # Streamed, not buffered.
        self.assertTrue(received["is_generator"])
        self.assertEqual(received["pulled_after_first_expert"], 2)

        self.assertEqual(
            seen_experts,
            [
                "model.layers.0.mlp.experts.gate_up_proj_blocks",
                "model.layers.0.mlp.experts.down_proj_blocks",
            ],
        )

        self.assertEqual(
            normal_seen["names"],
            [
                "model.embed_tokens.weight",
                "model.layers.0.input_layernorm.weight",
                "lm_head.weight",
            ],
        )
        self.assertEqual(normal_seen["other"], {"loaded_expert_param"})


class TestGptOssMxfp4ExpertRouting(unittest.TestCase):
    """Each expert tensor routes on its own name.

    The layout used to be decided by one ``any()`` probe over the whole list.
    Streaming removed the list, so a probe would consume the stream; latching
    the decision on the first tensor instead would send every tensor of the
    other layout down a branch that matches none of its names and drop it with
    no error.
    """

    HIDDEN = 64
    INTERMEDIATE = 32
    MXFP4_BLOCK = 32
    NUM_EXPERTS = 2

    def _model(self):
        base = "model.layers.0.mlp.experts."
        h, i, n = self.HIDDEN, self.INTERMEDIATE, self.NUM_EXPERTS
        params = {
            base + "w13_weight": torch.zeros(n, 2 * i, h // 2, dtype=torch.uint8),
            base
            + "w13_weight_scale": torch.zeros(
                n, 2 * i, h // self.MXFP4_BLOCK, dtype=torch.uint8
            ),
            base + "w13_weight_bias": torch.zeros(n, 2 * i, dtype=torch.bfloat16),
            base + "w2_weight": torch.zeros(n, h, i // 2, dtype=torch.uint8),
            base
            + "w2_weight_scale": torch.zeros(
                n, h, i // self.MXFP4_BLOCK, dtype=torch.uint8
            ),
            base + "w2_weight_bias": torch.zeros(n, h, dtype=torch.bfloat16),
        }
        moe = types.SimpleNamespace(tp_rank=0, tp_size=1, ep_rank=0, ep_size=1)
        model = types.SimpleNamespace(
            named_parameters=lambda: params.items(),
            mapping=types.SimpleNamespace(moe=moe),
            config=types.SimpleNamespace(
                intermediate_size=i, num_local_experts=n, hidden_size=h
            ),
        )
        model._load_mxfp4_per_expert_weights = (
            lambda *a, **k: GptOssForCausalLM._load_mxfp4_per_expert_weights(
                model, *a, **k
            )
        )
        return model, params, base

    def _per_expert_stream(self, base):
        """AMD-Quark tensors, preceded by a name that matches neither layout."""
        h, i = self.HIDDEN, self.INTERMEDIATE
        yield base + "format_marker", torch.zeros(1)
        for e in range(self.NUM_EXPERTS):
            p = f"{base}{e}."
            yield p + "gate_up_proj.weight", torch.zeros(
                2 * i, h // 2, dtype=torch.uint8
            )
            yield p + "gate_up_proj.weight_scale", torch.zeros(
                2 * i, h // self.MXFP4_BLOCK, dtype=torch.uint8
            )
            yield p + "gate_up_proj.bias", torch.zeros(2 * i, dtype=torch.bfloat16)
            yield p + "down_proj.weight", torch.zeros(h, i // 2, dtype=torch.uint8)
            yield p + "down_proj.weight_scale", torch.zeros(
                h, i // self.MXFP4_BLOCK, dtype=torch.uint8
            )
            yield p + "down_proj.bias", torch.zeros(h, dtype=torch.bfloat16)

    def test_per_expert_layout_loads_when_a_foreign_name_comes_first(self):
        model, params, base = self._model()

        loaded = GptOssForCausalLM._load_mxfp4_experts_weights(
            model, self._per_expert_stream(base)
        )

        # Latching the layout off the leading ``format_marker`` would be empty.
        self.assertEqual(loaded, set(params))


if __name__ == "__main__":
    unittest.main()
