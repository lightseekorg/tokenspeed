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

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.models.gpt_oss import GptOssForCausalLM


class TestGptOssMxfp4Streaming(unittest.TestCase):
    def test_load_mxfp4_weights_streams_experts(self):
        # Expert tensors interleaved with the small non-expert weights, in
        # checkpoint order. The "weight" stand-in is just the name string,
        # because the stubbed loaders below never touch tensor data.
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
            # The dispatcher must hand us a lazy generator, not a materialized
            # list of every expert tensor.
            received["is_generator"] = isinstance(weights, types.GeneratorType)
            iterator = iter(weights)
            first_expert = next(iterator)
            seen_experts.append(first_expert[0])
            # Reaching the first expert (item #2) must not have drained the
            # whole source iterator (5 items) -- proof that loading is
            # interleaved with iteration, i.e. streamed.
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

        # Expert tensors (matched by the ".experts" marker) are routed to the
        # expert loader, in order.
        self.assertEqual(
            seen_experts,
            [
                "model.layers.0.mlp.experts.gate_up_proj_blocks",
                "model.layers.0.mlp.experts.down_proj_blocks",
            ],
        )

        # Everything else is collected for the generic loader, and the set of
        # already-loaded expert params is threaded through to it.
        self.assertEqual(
            normal_seen["names"],
            [
                "model.embed_tokens.weight",
                "model.layers.0.input_layernorm.weight",
                "lm_head.weight",
            ],
        )
        self.assertEqual(normal_seen["other"], {"loaded_expert_param"})


if __name__ == "__main__":
    unittest.main()
