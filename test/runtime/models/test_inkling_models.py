"""Inkling model unit tests: config registration, scheduler blindness, fixtures.

Model-module tests (gate math, sconv parity, shapes) are added alongside the
model implementation. NOTE: intentionally NOT registered in CI suites while
the Inkling port is confidential/local-only.
"""

import os
import sys
import tempfile
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
from test.runtime.models.inkling_fixtures import (
    TINY_MM_CONFIG,
    make_inkling_dummy_checkpoint,
)


class TestInklingConfigRegistry(unittest.TestCase):
    def test_config_registry(self):
        self.assertIs(_CONFIG_REGISTRY["inkling_mm_model"], InklingMMConfig)
        self.assertIs(_CONFIG_REGISTRY["inkling_model"], InklingModelConfig)

    def test_get_config_loads_tiny_fixture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = make_inkling_dummy_checkpoint(tmpdir, tiny=True)
            config = get_config(str(ckpt), trust_remote_code=False, revision=None)
        self.assertIsInstance(config, InklingMMConfig)
        text = config.get_text_config()
        self.assertEqual(text.num_hidden_layers, 6)
        self.assertEqual(text.swa_attention_layer_ids, [0, 1, 2, 3, 4])
        self.assertEqual(text.global_attention_layer_ids, [5])

    def test_scheduler_blindness(self):
        """The engine enables mamba scheduling off attribute presence alone
        (event_loop.py); the Inkling config must never grow these attributes."""
        config = InklingMMConfig(**{k: v for k, v in TINY_MM_CONFIG.items()})
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
        text = InklingModelConfig(**TINY_MM_CONFIG["text_config"])
        self.assertEqual(text.ckpt_num_key_value_heads, 2)
        self.assertEqual(text.num_key_value_heads, 4)

    def test_vocab_semantics(self):
        text = InklingModelConfig(**TINY_MM_CONFIG["text_config"])
        self.assertEqual(text.vocab_size, 2000)  # unpadded, for reporting
        self.assertEqual(text.padded_vocab_size, 2048)  # module shapes

    def test_conv_stream_layout(self):
        text = InklingModelConfig(**TINY_MM_CONFIG["text_config"])
        layout = inkling_conv_stream_layout(text, attn_tp_size=1)
        kv_dim = 4 * 32  # uniform kv heads * head_dim
        self.assertEqual(layout[InklingConvStream.K], (0, kv_dim))
        self.assertEqual(layout[InklingConvStream.V], (kv_dim, kv_dim))
        self.assertEqual(layout[InklingConvStream.ATTN], (2 * kv_dim, 256))
        self.assertEqual(layout[InklingConvStream.MLP], (2 * kv_dim + 256, 256))
        self.assertEqual(inkling_conv_total_dim(text, 1), 2 * kv_dim + 2 * 256)

    def test_paged_cache_layer_types_sliding_subgroups(self):
        """Flat-KV group labels: sliding layers split round-robin into
        equal-count sub-groups sized by the full-layer count, so every
        hybrid slab is bound by one layer of each group (5+1 for Inkling;
        see paged_cache_spec.hybrid_slab_group_size). Exposed as
        paged_cache_layer_types, NOT layer_types: transformers validates
        layer_types against ALLOWED_LAYER_TYPES, which rejects sub-group
        labels."""
        text = InklingModelConfig(**TINY_MM_CONFIG["text_config"])
        self.assertEqual(
            text.paged_cache_layer_types,
            [f"sliding_attention_{k}" for k in range(5)] + ["full_attention"],
        )
        # Real shape: 66 layers, full attention every 6th layer ->
        # 5 sub-groups of 11 sliding + 11 full.
        full_ids = list(range(5, 66, 6))
        cfg = InklingModelConfig(
            num_hidden_layers=66,
            local_layer_ids=[i for i in range(66) if i not in set(full_ids)],
        )
        lt = cfg.paged_cache_layer_types
        self.assertEqual(
            [i for i, t in enumerate(lt) if t == "full_attention"], full_ids
        )
        self.assertEqual(lt.count("full_attention"), 11)
        for k in range(5):
            self.assertEqual(lt.count(f"sliding_attention_{k}"), 11)
        # Draft (nextn) shape: local_layer_ids=[] -> all full, indexable
        # by draft layer_id.
        draft = InklingModelConfig(num_hidden_layers=4, local_layer_ids=[])
        self.assertEqual(draft.paged_cache_layer_types, ["full_attention"] * 4)

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
        text = InklingModelConfig(**TINY_MM_CONFIG["text_config"])
        gate = InklingGate(text).cuda()
        with torch.no_grad():
            gate.weight.normal_(0, 0.5)
            gate.bias.normal_(0, 0.1)
            # Identity scale keeps the torch reference below exact.
            gate.global_scale.fill_(1.0)
        x = torch.randn(64, text.hidden_size, device="cuda")

        full_weights, ids, logits = gate(x)
        weights, gammas = full_weights[:, : gate.top_k], full_weights[:, gate.top_k :]
        ref_w, ref_ids, ref_g = _reference_gate(
            x.cpu(), gate.weight.cpu(), gate.bias.cpu(), 8, 2, 2, text.route_scale
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
        text = InklingModelConfig(**TINY_MM_CONFIG["text_config"])
        mapping = SimpleNamespace(attn=SimpleNamespace(tp_rank=0, tp_size=1))
        mod = InklingSharedExperts(text, mapping).to(torch.bfloat16).cuda()
        with torch.no_grad():
            mod.w13_weight.normal_(0, 0.05)
            mod.w2_weight.normal_(0, 0.05)

        x = torch.randn(37, text.hidden_size, dtype=torch.bfloat16, device="cuda")
        gammas = torch.rand(37, text.n_shared_experts, device="cuda") * 8

        finalized = mod(x, gammas)
        deferred = mod(x, do_finalize=False)
        self.assertEqual(
            tuple(deferred.shape), (text.n_shared_experts, 37, text.hidden_size)
        )
        manual = torch.einsum("sth,ts->th", deferred.float(), gammas.float())
        torch.testing.assert_close(finalized.float(), manual, atol=2e-2, rtol=2e-2)


def _ref_sconv(x, weight, prefix, use_residual=True):
    """Torch reference: residual causal FIR, current token = last tap."""
    import torch

    W = weight.shape[1]
    xp = torch.cat([prefix.float(), x.float()])
    y = sum(xp[w : w + len(x)] * weight[:, w].float() for w in range(W))
    return (y + x.float()) if use_residual else y


@unittest.skipUnless(__import__("torch").cuda.is_available(), "sconv kernels need CUDA")
class TestInklingShortConvolution(unittest.TestCase):
    def _make_ctx(self, num_layers, num_slots, conv_dim, kernel_size, device):
        from types import SimpleNamespace

        import torch

        from tokenspeed.runtime.layers.attention.backends.inkling import (
            InklingAttnBackend,
            InklingConvStatePool,
        )

        pool = InklingConvStatePool(
            num_layers, num_slots, conv_dim, kernel_size, torch.bfloat16, device
        )
        backend = SimpleNamespace(conv_pool=pool, conv_metadata=None)
        # The real window-persist method; stateless in the default inplace
        # mode, so binding it to the namespace mock is faithful.
        backend.apply_conv_state_update = (
            InklingAttnBackend.apply_conv_state_update.__get__(backend)
        )
        ctx = SimpleNamespace(attn_backend=backend)
        return ctx, backend, pool

    def test_prefill_then_decode_parity(self):
        import torch

        from tokenspeed.runtime.configs.inkling_config import InklingConvStream
        from tokenspeed.runtime.layers.attention.backends.inkling import (
            InklingConvMetadata,
        )
        from tokenspeed.runtime.models.inkling import InklingShortConvolution

        torch.manual_seed(0)
        device = "cuda"
        dim, W, offset = 64, 4, 32
        conv_dim = offset + dim + 16  # sconv operates on a channel slice
        ctx, backend, pool = self._make_ctx(1, 4, conv_dim, W, device)

        mod = InklingShortConvolution(
            dim, W, InklingConvStream.K, layer_id=0, channel_offset=offset
        ).to(device)
        with torch.no_grad():
            mod.weight.normal_(0, 0.5)
        w2d = mod.weight.squeeze(1)

        # Prefill: two sequences, no initial state.
        lens = [13, 7]
        total = sum(lens)
        x = torch.randn(total, dim, device=device, dtype=torch.bfloat16)
        cu = torch.tensor([0, 13, 20], dtype=torch.int32, device=device)
        from tokenspeed_kernel.ops.conv import seq_idx_from_cu_seqlens

        backend.conv_metadata = InklingConvMetadata(
            query_start_loc=cu,
            cache_indices=torch.tensor([1, 2], dtype=torch.int32, device=device),
            has_initial_state=torch.zeros(2, dtype=torch.bool, device=device),
            is_decode=False,
            seq_idx=seq_idx_from_cu_seqlens(cu, total),
        )
        y = mod(x, ctx)
        zeros = torch.zeros(W - 1, dim, device=device)
        for i, (s, e) in enumerate(zip(cu[:-1].tolist(), cu[1:].tolist())):
            ref = _ref_sconv(x[s:e], w2d, zeros)
            torch.testing.assert_close(y[s:e].float(), ref, atol=2e-2, rtol=2e-2)

        # Decode 3 steps: parity vs continuing the same sequences.
        full_x = [x[:13], x[13:]]
        for _step in range(3):
            xt = torch.randn(2, dim, device=device, dtype=torch.bfloat16)
            backend.conv_metadata = InklingConvMetadata(
                query_start_loc=torch.arange(3, dtype=torch.int32, device=device),
                cache_indices=torch.tensor([1, 2], dtype=torch.int32, device=device),
                has_initial_state=torch.ones(2, dtype=torch.bool, device=device),
                is_decode=True,
            )
            yt = mod(xt, ctx)
            for i in range(2):
                seq = torch.cat([full_x[i], xt[i : i + 1]])
                ref_full = _ref_sconv(seq, w2d, torch.zeros(W - 1, dim, device=device))
                torch.testing.assert_close(
                    yt[i].float(), ref_full[-1], atol=3e-2, rtol=3e-2
                )
                full_x[i] = seq

        # Channels outside the slice untouched.
        state = pool.layer_state(0)
        self.assertTrue((state[:, :offset, :] == 0).all())
        self.assertTrue((state[:, offset + dim :, :] == 0).all())


if __name__ == "__main__":
    unittest.main()
