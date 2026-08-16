"""Inkling model unit tests: config registration, scheduler blindness, modules.

All configs come from the released hub snapshots (``inkling_fixtures``),
truncated in layer count only — module tests run at the real widths, on the
real kernel paths (fused router GEMM etc.). Model-module tests (gate math,
sconv parity, shapes) are added alongside the model implementation.
"""

import os
import sys
import unittest

from tokenspeed.runtime.configs.inkling_config import (
    InklingConvStream,
    InklingMMConfig,
    InklingModelConfig,
    inkling_conv_stream_layout,
    inkling_conv_total_dim,
)
from tokenspeed.runtime.utils.hf_transformers_utils import _CONFIG_REGISTRY, get_config

# Add project root directory to path for importing test.* helpers.
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from test.runtime.models.inkling_fixtures import (  # noqa: E402
    INKLING_NVFP4,
    NUM_TEST_LAYERS,
    has_blackwell,
    load_inkling_config,
)

from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(
    est_time=40,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*", "h100-*"],
)

if not has_blackwell():
    raise unittest.SkipTest("Inkling module tests require an NVIDIA Blackwell GPU")


class TestInklingConfigRegistry(unittest.TestCase):
    def test_config_registry(self):
        self.assertIs(_CONFIG_REGISTRY["inkling_mm_model"], InklingMMConfig)
        self.assertIs(_CONFIG_REGISTRY["inkling_model"], InklingModelConfig)

    def test_get_config_loads_hub_snapshot(self):
        config = get_config(INKLING_NVFP4, trust_remote_code=False, revision=None)
        self.assertIsInstance(config, InklingMMConfig)
        text = config.get_text_config()
        self.assertEqual(text.num_hidden_layers, 66)
        # Full attention every 6th layer; the rest sliding.
        self.assertEqual(text.global_attention_layer_ids, list(range(5, 66, 6)))
        self.assertEqual(
            text.swa_attention_layer_ids,
            [i for i in range(66) if i % 6 != 5],
        )

    def test_truncated_config_keeps_all_layer_kinds(self):
        text = load_inkling_config().get_text_config()
        self.assertEqual(text.num_hidden_layers, NUM_TEST_LAYERS)
        self.assertEqual(text.dense_mlp_idx, 2)  # dense layers present
        self.assertEqual(text.swa_attention_layer_ids, [0, 1, 2, 3, 4])
        self.assertEqual(text.global_attention_layer_ids, [5])

    def test_scheduler_blindness(self):
        """The engine enables mamba scheduling off attribute presence alone
        (event_loop.py); the Inkling config must never grow these attributes."""
        config = load_inkling_config()
        text = config.get_text_config()
        for obj in (config, text):
            for attr in (
                "mamba2_cache_params",
                "mamba_chunk_size",
                "conv_layer_ids",
                "linear_layer_ids",
                "full_attention_layer_ids",
            ):
                self.assertFalse(
                    hasattr(obj, attr),
                    f"{type(obj).__name__}.{attr} present: engages hybrid/"
                    "mamba engine paths; Inkling must look like dense GQA.",
                )

    def test_kv_head_uniformization(self):
        text = load_inkling_config().get_text_config()
        self.assertEqual(text.ckpt_num_key_value_heads, 8)
        self.assertEqual(text.swa_num_key_value_heads, 16)
        self.assertEqual(text.num_key_value_heads, 16)

    def test_vocab_semantics(self):
        text = load_inkling_config().get_text_config()
        self.assertEqual(text.vocab_size, 200058)  # unpadded, for reporting
        self.assertEqual(text.padded_vocab_size, 201024)  # module shapes

    def test_conv_stream_layout(self):
        text = load_inkling_config().get_text_config()
        layout = inkling_conv_stream_layout(text, attn_tp_size=1)
        kv_dim = text.num_key_value_heads * text.head_dim  # uniform kv width
        h = text.hidden_size
        self.assertEqual(layout[InklingConvStream.K], (0, kv_dim))
        self.assertEqual(layout[InklingConvStream.V], (kv_dim, kv_dim))
        self.assertEqual(layout[InklingConvStream.ATTN], (2 * kv_dim, h))
        self.assertEqual(layout[InklingConvStream.MLP], (2 * kv_dim + h, h))
        self.assertEqual(inkling_conv_total_dim(text, 1), 2 * kv_dim + 2 * h)

    def test_cache_layer_types_sliding_subgroups(self):
        """Cache-group labels: sliding layers split round-robin into
        equal-count sub-groups sized by the full-layer count, so every
        hybrid slab is bound by one layer of each group (5+1 truncated,
        5x11+11 at full depth; see kv_cache.recipes.publish.hybrid_slab_group_size).
        Exposed as cache_layer_types, NOT layer_types: transformers
        validates layer_types against ALLOWED_LAYER_TYPES, which rejects
        sub-group labels."""
        text = load_inkling_config().get_text_config()
        self.assertEqual(
            text.cache_layer_types,
            [f"sliding_attention_{k}" for k in range(5)] + ["full_attention"],
        )
        # Full depth: 66 layers -> 5 sub-groups of 11 sliding + 11 full.
        full = load_inkling_config(num_layers=None).get_text_config()
        full_ids = list(range(5, 66, 6))
        lt = full.cache_layer_types
        self.assertEqual(
            [i for i, t in enumerate(lt) if t == "full_attention"], full_ids
        )
        self.assertEqual(lt.count("full_attention"), 11)
        for k in range(5):
            self.assertEqual(lt.count(f"sliding_attention_{k}"), 11)
        # Draft (nextn) shape: local_layer_ids=[] -> all full, indexable
        # by draft layer_id.
        draft = InklingModelConfig(num_hidden_layers=4, local_layer_ids=[])
        self.assertEqual(draft.cache_layer_types, ["full_attention"] * 4)

    def test_local_layer_ids_validation(self):
        with self.assertRaises(ValueError):
            InklingModelConfig(num_hidden_layers=4, local_layer_ids=[0, 9])
        with self.assertRaises(ValueError):
            InklingModelConfig(num_hidden_layers=4, local_layer_ids=[0, 0])


def _reference_gate(x, weight, bias, n_routed, n_shared, top_k, route_scale):
    """Independent port of the reference InklingGate math (sigmoid path)."""
    import torch
    import torch.nn.functional as F

    logits = (x.float() @ weight.t().float()).float()
    routed = logits[:, :n_routed]
    sel = routed.sigmoid()
    if bias is not None:
        sel = sel + bias
    _, ids = torch.topk(sel, top_k, dim=-1)
    active = torch.cat([routed.gather(-1, ids), logits[:, n_routed:]], dim=-1)
    logp = F.logsigmoid(active)
    w = torch.exp(logp - torch.logsumexp(logp, dim=-1, keepdim=True)) * route_scale
    return w[:, :top_k], ids, w[:, top_k:]


@unittest.skipUnless(
    __import__("torch").cuda.is_available(), "fused gate kernel needs CUDA"
)
class TestInklingGate(unittest.TestCase):
    def test_gate_math_matches_reference(self):
        import torch

        from tokenspeed.runtime.models.inkling import InklingGate

        torch.manual_seed(0)
        text = load_inkling_config().get_text_config()
        # bf16 weight as in the engine: the real hidden size takes the fused
        # router-GEMM path, which requires bf16 activations and weight.
        torch.set_default_dtype(torch.bfloat16)
        try:
            gate = InklingGate(text).cuda()
        finally:
            torch.set_default_dtype(torch.float32)
        self.assertTrue(gate.use_dsv3_router_gemm)
        with torch.no_grad():
            gate.weight.normal_(0, 0.5)
            gate.bias.normal_(0, 0.1)
            # Identity scale keeps the torch reference below exact.
            gate.global_scale.fill_(1.0)
        x = torch.randn(64, text.hidden_size, dtype=torch.bfloat16, device="cuda")

        full_weights, ids, logits = gate(x)
        weights, gammas = full_weights[:, : gate.top_k], full_weights[:, gate.top_k :]
        ref_w, ref_ids, ref_g = _reference_gate(
            x.cpu(),
            gate.weight.cpu(),
            gate.bias.cpu(),
            text.n_routed_experts,
            text.n_shared_experts,
            gate.top_k,
            text.route_scale,
        )
        weights, gammas, ids = weights.cpu(), gammas.cpu(), ids.cpu()
        self.assertTrue(torch.equal(ids.long(), ref_ids))
        torch.testing.assert_close(weights, ref_w, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(gammas, ref_g, atol=1e-5, rtol=1e-5)
        # Weights across selected-routed + shared sum to route_scale.
        total = weights.sum(-1) + gammas.sum(-1)
        torch.testing.assert_close(
            total, torch.full_like(total, text.route_scale), atol=1e-4, rtol=1e-4
        )


@unittest.skipUnless(
    __import__("torch").cuda.is_available(), "SiluAndMul kernel needs CUDA"
)
class TestInklingSharedExperts(unittest.TestCase):
    def test_deferred_matches_finalized(self):
        """do_finalize=False returns un-weighted [S, T, H]; weighting it with
        the gammas must reproduce the finalized [T, H] output."""
        from types import SimpleNamespace

        import torch

        from tokenspeed.runtime.models.inkling import InklingSharedExperts

        torch.manual_seed(0)
        text = load_inkling_config().get_text_config()
        mapping = SimpleNamespace(attn=SimpleNamespace(tp_rank=0, tp_size=1))
        mod = InklingSharedExperts(text, mapping).to(torch.bfloat16).cuda()
        with torch.no_grad():
            # Fan-in-scaled init keeps activations O(1) at the real widths, so
            # the bf16 kernel/einsum comparison stays inside the tolerance.
            mod.w13_weight.normal_(0, text.hidden_size**-0.5)
            mod.w2_weight.normal_(0, text.intermediate_size**-0.5)

        x = torch.randn(37, text.hidden_size, dtype=torch.bfloat16, device="cuda")
        gammas = torch.rand(37, text.n_shared_experts, device="cuda") * 8

        finalized = mod(x, gammas)
        deferred = mod(x, do_finalize=False)
        self.assertEqual(
            tuple(deferred.shape), (text.n_shared_experts, 37, text.hidden_size)
        )
        manual = torch.einsum("sth,ts->th", deferred.float(), gammas.float())
        # The finalized branch folds gammas into bf16 intermediates BEFORE the
        # down-proj (linear, so mathematically equivalent); at the real widths
        # that pre- vs post-weighting rounds differently, so tolerate bf16
        # noise rather than near-equality.
        torch.testing.assert_close(finalized.float(), manual, atol=2e-1, rtol=5e-2)


if __name__ == "__main__":
    unittest.main()
